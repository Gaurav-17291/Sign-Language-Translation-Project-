import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

import utils
from utils.rec_net_mp import SLRModel, compute_lgt
from dataloader.feeder_mp import BaseFeeder

class ExtractorWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, vid_full, vid_face, vid_lh, vid_rh, len_x):
        # Running the normal forward pass
        ret_dict = self.base_model(vid_full, vid_face, vid_lh, vid_rh, len_x)
        return ret_dict["conv1d_features"].transpose(0, 1)

def main():
    # --- PATHS ---
    WEIGHTS_PATH = "./work_dir/Phase1_Resnet_mp/_best_model.pt"
    OUTPUT_TRAIN_DIR = "./work_dir/Phase1_Resnet_mp/extracted_conv1d_features/train"
    OUTPUT_DEV_DIR = "./work_dir/Phase1_Resnet_mp/extracted_conv1d_features/dev"
    OUTPUT_TEST_DIR = "./work_dir/Phase1_Resnet_mp/extracted_conv1d_features/test"

    os.makedirs(OUTPUT_TRAIN_DIR, exist_ok=True)
    os.makedirs(OUTPUT_TEST_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DEV_DIR, exist_ok=True)

    # LOAD DICTIONARY
    DICT_PATH = "./datainfo/gloss_dict.npy"
    gloss_dict = np.load(DICT_PATH, allow_pickle=True).item()
    NUM_CLASSES = len(gloss_dict) + 1

    # LOAD MULTI-CUE MODEL
    print("1. Loading Multi-Cue SLR Model...")
    base_model = SLRModel(num_classes=NUM_CLASSES, hidden_size=512, gloss_dict=gloss_dict)

    checkpoint = torch.load(WEIGHTS_PATH, map_location='cpu', weights_only=False)

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v

    base_model.load_state_dict(clean_state_dict, strict=True)
    base_model.eval() 
    print("Weights loaded successfully!")

    wrapper = ExtractorWrapper(base_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = wrapper.to(device)

    if torch.cuda.device_count() > 1:
        print(f" Unlocking {torch.cuda.device_count()} GPUs for parallel extraction")
        model = torch.nn.DataParallel(model)

    # --- EXTRACTOR LOOP ---
    def extract_split(mode, output_dir):
        print(f"\n2. Extracting {mode.upper()} split...")
        dataset = BaseFeeder(
            prefix="/dev/shm/md2409_dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-256x256px",
            gloss_dict=gloss_dict,
            mode=mode,
            transform_mode=False
        )

        loader = DataLoader(
            dataset, batch_size=2, shuffle=False,
            num_workers=4, collate_fn=dataset.collate_fn,
            pin_memory=True
        )

        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(loader)):
                vid_full = data[0].to(device)
                vid_face = data[1].to(device)
                vid_lh = data[2].to(device)
                vid_rh = data[3].to(device)
                len_x = data[4].to(device)
                info = data[-1] # Video folder names

                ret_features = model(vid_full, vid_face, vid_lh, vid_rh, len_x)

                conv1d_feats = ret_features.cpu().numpy()

                valid_lengths = compute_lgt(len_x, base_model.cond1d_type, base_model.cond1d_size).numpy()

                for i, vid_name in enumerate(info):
                    vid_id = vid_name.split("|")[0]
                    valid_len = int(valid_lengths[i])

                    clean_feat = conv1d_feats[i, :valid_len, :]

                    save_path = os.path.join(output_dir, f"{vid_id}.npy")
                    np.save(save_path, clean_feat)

    extract_split("train", OUTPUT_TRAIN_DIR)
    extract_split("dev", OUTPUT_DEV_DIR)
    extract_split("test", OUTPUT_TEST_DIR)
    print("\nExtraction Complete! Multi-Cue features are perfectly shaped for mBART.")

if __name__ == "__main__":
    main()
