import os
import yaml
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = False
cudnn.deterministic = True
import importlib
import faulthandler
import numpy as np
import torch.nn as nn
from tqdm import tqdm

# DDP IMPORTS
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

faulthandler.enable()

import utils
from Evaluation.wer_calculation import evaluate

class ModelController():
    def __init__(self, arg, local_rank):
        self.arg = arg
        self.local_rank = local_rank
        if self.local_rank == 0:
            self.save_arg()

        if self.arg.random_fix:
            self.rng = utils.RandomState(seed=self.arg.random_seed)

        self.recoder = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval) if self.local_rank == 0 else None

        self.dataset = {}
        self.data_loader = {}

        self.gloss_dict = np.load(self.arg.dataset_info['dict_path'], allow_pickle=True).item()
        self.arg.model_args['num_classes'] = len(self.gloss_dict) + 1

        self.model = self.load_model()
        self.optimizer = utils.Optimizer(self.model, self.arg.optimizer_args)
        self.model, self.optimizer = self.model_to_device(self.model, self.optimizer)

    def run(self):
        if self.arg.phase == 'train':
            if self.local_rank == 0:
                self.recoder.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))

            seq_model_list = []
            best_dev = 100.0
            best_epoch = 0

            for epoch in range(self.arg.optimizer_args['start_epoch'], self.arg.num_epoch):
                save_model = epoch % self.arg.save_interval == 0
                dev_flag = epoch % 1 == 0

                # Shuffle the DDP dataloader every epoch
                if 'WORLD_SIZE' in os.environ:
                    self.data_loader['train'].sampler.set_epoch(epoch)

                train_model(self.data_loader['train'], self.model, self.optimizer,
                            self.local_rank, epoch, self.recoder)

                # Evaluate and Save ONLY on Master GPU
                if self.local_rank == 0 and dev_flag:
                    dev_wer = eval_model(self.arg, self.data_loader['dev'], self.model, self.local_rank,
                                         'dev', epoch, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
                    self.recoder.print_log("Dev WER: {:05.2f}%".format(dev_wer))

                    if dev_wer < best_dev:
                        best_dev = dev_wer
                        best_epoch = epoch
                        model_path = "{}_best_model.pt".format(self.arg.work_dir)
                        self.save_model(epoch, model_path)
                        self.recoder.print_log('Save best model')

                    self.recoder.print_log('Best_dev: {:05.2f}, Epoch : {}'.format(best_dev, best_epoch))

                    if save_model:
                        model_path = "{}dev_{:05.2f}_epoch{}_model.pt".format(self.arg.work_dir, dev_wer, epoch)
                        seq_model_list.append(model_path)
                        self.save_model(epoch, model_path)

                if 'WORLD_SIZE' in os.environ:
                    dist.barrier()

        elif self.arg.phase == 'test':
            if self.local_rank == 0:
                self.recoder.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
                self.recoder.print_log('--- Starting Test Set Evaluation ---')

            test_wer = eval_model(self.arg, self.data_loader['test'], self.model, self.local_rank,
                                  'test', 40, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
            
            if self.local_rank == 0:
                self.recoder.print_log("======================================")
                self.recoder.print_log("Final Test WER: {:05.2f}%".format(test_wer))
                self.recoder.print_log("======================================")

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def save_model(self, epoch, save_path):
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.optimizer.scheduler.state_dict(),
            'rng_state': self.rng.save_rng_state(),
        }, save_path)

    def load_model(self):
        if self.local_rank == 0:
            print("Loading model")
        model_class = import_class(self.arg.model)
        model = model_class(
            **self.arg.model_args,
            gloss_dict=self.gloss_dict,
            loss_weights=self.arg.loss_weights,
        )
        return model

    def model_to_device(self, model, optimizer):
        if self.arg.load_weights:
            self.load_model_weights(model, self.arg.load_weights)
        elif self.arg.load_checkpoints:
            self.load_checkpoint_weights(model, optimizer)

        self.load_data()

        model = model.cuda(self.local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        if 'WORLD_SIZE' in os.environ:
            model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=False)

        if self.local_rank == 0:
            print("Loading model finished.")
        return model, optimizer

    def load_model_weights(self, model, weight_path):
        state_dict = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(state_dict['model_state_dict'], strict=False)

    def load_checkpoint_weights(self, model, optimizer):
        self.load_model_weights(model, self.arg.load_checkpoints)
        state_dict = torch.load(self.arg.load_checkpoints, map_location='cpu')
        if self.local_rank == 0:
            print("Loading ckpt start!")
        
        if 'rng_state' in state_dict:
            state_dict['rng_state']['torch'] = state_dict['rng_state']['torch'].cpu()

        if len(torch.cuda.get_rng_state_all()) == len(state_dict['rng_state']['cuda']):
            self.rng.set_rng_state(state_dict['rng_state'])
        
        if "optimizer_state_dict" in state_dict.keys():
            optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            for state in optimizer.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(self.local_rank)
        
        if "scheduler_state_dict" in state_dict.keys():
            optimizer.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        self.arg.optimizer_args['start_epoch'] = state_dict["epoch"] + 1

    def load_data(self):
        if self.local_rank == 0:
            print("Loading data")
        self.feeder = import_class(self.arg.feeder)
        dataset_list = zip(["train", "train_eval", "dev", "test"], [True, False, False, False])
        for idx, (mode, train_flag) in enumerate(dataset_list):
            arg = self.arg.feeder_args
            arg["prefix"] = self.arg.dataset_info['dataset_root']
            arg["mode"] = mode.split("_")[0]
            arg["transform_mode"] = train_flag
            self.dataset[mode] = self.feeder(gloss_dict=self.gloss_dict, **arg)

            # DDP DATALOADER SETUP
            if train_flag and 'WORLD_SIZE' in os.environ:
                sampler = DistributedSampler(self.dataset[mode])
                shuffle = False
            else:
                sampler = None
                shuffle = train_flag

            self.data_loader[mode] = torch.utils.data.DataLoader(
                self.dataset[mode],
                batch_size=self.arg.batch_size if mode == "train" else self.arg.test_batch_size,
                shuffle=shuffle,
                sampler=sampler,
                drop_last=train_flag,
                num_workers=self.arg.num_worker,
                collate_fn=self.feeder.collate_fn,
                pin_memory=True,
                persistent_workers=True if self.arg.num_worker > 0 else False
            )

def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod

def train_model(loader, model, optimizer, local_rank, epoch_idx, recoder):
    model.train()
    loss_value = []
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]

    pbar = tqdm(loader) if local_rank == 0 else loader

    for batch_idx, data in enumerate(pbar):
        vid = data[0].cuda(local_rank)
        vid_lgt = data[1].cuda(local_rank)
        label = data[2].cuda(local_rank)
        label_lgt = data[3].cuda(local_rank)
        ann = data[4]

        optimizer.zero_grad()

        ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt, ann=ann)

        with torch.backends.cudnn.flags(enabled=False):
            loss_func = model.module.losses_calculation if hasattr(model, 'module') else model.losses_calculation
            loss_dict = loss_func(ret_dict, label, label_lgt.cpu())

        loss = loss_dict["total_loss"]
        loss = loss.mean()

        if np.isinf(loss.item()) or np.isnan(loss.item()):
            raise ValueError(f"CRASH: GPU {local_rank} got NaN loss for {data[-1]}")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.optimizer.step()

        loss_value.append(loss.item())

        if local_rank == 0 and batch_idx % (recoder.log_interval if recoder else 100) == 0:
            recoder.print_log(
                'Epoch: {}, Batch({}/{}) done. Loss: {:.5f} lr:{:.7f}'.format(
                    epoch_idx, batch_idx, len(loader), loss.item(), clr[0])
            )

    optimizer.scheduler.step()
    if local_rank == 0:
        recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))
    return loss_value

def eval_model(cfg, loader, model, local_rank, mode, epoch, work_dir, recoder, evaluate_tool="python"):
    model.eval()
    total_sent = []
    total_info = []
    
    pbar = tqdm(loader) if local_rank == 0 else loader
    
    for batch_idx, data in enumerate(pbar):
        vid = data[0].cuda(local_rank)
        vid_lgt = data[1].cuda(local_rank)
        label = data[2].cuda(local_rank)
        label_lgt = data[3].cuda(local_rank)
        
        with torch.no_grad():
            eval_model = model.module if hasattr(model, 'module') else model
            ret_dict = eval_model(vid, vid_lgt, label=label, label_lgt=label_lgt)
            
        total_info += [file_name.split("|")[0] for file_name in data[-1]]
        total_sent += ret_dict['recognized_sents']

    try:
        write2file(work_dir + "/output-hypothesis-{}.ctm".format(mode), total_info, total_sent)
        lstm_ret = evaluate(
            prefix=work_dir, mode=mode, output_file="output-hypothesis-{}.ctm".format(mode),
            evaluate_dir=cfg.dataset_info['evaluate_dir'],
            evaluate_prefix=cfg.dataset_info['evaluate_prefix'],
            output_dir="epoch_{}_result/".format(epoch),
            triplet=True,
        )
    except Exception as e:
        if local_rank == 0:
            print("Unexpected error in evaluation:", e)
        lstm_ret = 100.0

    if recoder and local_rank == 0:
        recoder.print_log(f"Epoch {epoch}, {mode} {lstm_ret: 2.2f}%", f"{work_dir}/{mode}.txt")
    return lstm_ret

def write2file(path, info, output):
    with open(path, "w") as filereader:
        for sample_idx, sample in enumerate(output):
            for word_idx, word in enumerate(sample):
                filereader.writelines(
                    "{} 1 {:.2f} {:.2f} {}\n".format(
                        info[sample_idx],
                        word_idx * 1.0 / 100,
                        (word_idx + 1) * 1.0 / 100,
                        word[0])
                )

if __name__ == '__main__':
    # Initialize DDP Network
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    parser = utils.get_parser()
    prs = parser.parse_args()

    with open(prs.config, 'r') as f:
        default_arg = yaml.load(f, Loader=yaml.FullLoader)
    parser.set_defaults(**default_arg)
    args = parser.parse_args()

    with open(f"./configs/datasetcfg.yaml", 'r') as f:
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)

    processor = ModelController(args, local_rank)
    processor.run()
