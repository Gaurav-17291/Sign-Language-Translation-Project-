import torch
import torch.optim as optim

class Optimizer(object):
    def __init__(self, model, optim_dict):
        self.optim_dict = optim_dict
        if self.optim_dict["optimizer"] == 'AdamW':
            alpha = self.optim_dict.get('learning_ratio', 1.0)
            weight_ratio = self.optim_dict.get('weight_ratio', 1.0)

            resnet_params = []
            scratch_params = []

            for n, p in model.named_parameters():
                # 1. CRITICAL SAFETY CHECK: Skip frozen parameters!
                if not p.requires_grad:
                    continue

                # 2. Sort into two distinct buckets
                if "resnet" in n:
                    resnet_params.append(p)
                else:
                    scratch_params.append(p)

            # 3. Apply the Differential Learning Rates
            self.optimizer = optim.AdamW(
                [
                    # Group 1: ResNet (Pretrained baseline, uses scaled learning rate 'alpha')
                    {'params': resnet_params, 'lr': self.optim_dict['base_lr'] * alpha, 'weight_decay': self.optim_dict['weight_decay'] * weight_ratio},

                    # Group 2: 1D CNN, BiLSTM, Classifier (Fast LR. Trained from scratch/fine-tuning)
                    {'params': scratch_params, 'lr': self.optim_dict['base_lr'], 'weight_decay': self.optim_dict['weight_decay']},
                ],
            )
        else:
            raise ValueError()
        self.scheduler = self.define_lr_scheduler(self.optimizer, self.optim_dict['step'])

    def define_lr_scheduler(self, optimizer, milestones):
        if self.optim_dict["optimizer"] in ['SGD', 'AdamW']:
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)
            return lr_scheduler
        else:
            raise ValueError()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def to(self, device):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
