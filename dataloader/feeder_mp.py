import os
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import sys
import glob
import torch
import warnings
import numpy as np
import torch.utils.data as data
from utils import video_augmentation

warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.append("..")

class BaseFeeder(data.Dataset):
    def __init__(self, prefix, gloss_dict, num_gloss=-1, mode="train", transform_mode=True):
        self.mode = mode
        self.ng = num_gloss
        self.prefix = prefix 
        self.dict = gloss_dict
        self.transform_mode = "train" if transform_mode else "test"
        
        info_path = f"./datainfo/{mode}_info.npy"
        self.inputs_list = np.load(info_path, allow_pickle=True).item()
        self.multicue_root = "/dev/shm/md2409_dataset/multicue_dataset"
        
        self.full_aug = self.transform_full()
        self.crop_aug = self.transform_crops()

    def __getitem__(self, idx):
        vid_full, vid_face, vid_lh, vid_rh, label, ann, fi = self.read_process(idx)
        
        # Note: We removed the [::2] from here! 
        # The striding is now done before the CPU wastes time reading the files.
        return vid_full, vid_face, vid_lh, vid_rh, torch.LongTensor(label), ann, fi['folder']

    def read_process(self, index):
        fi = self.inputs_list[index]
        video_name = fi['folder']
        
        full_frame_dir = os.path.join(self.prefix, self.mode, video_name)
        
        # 1. Grab the list of file paths
        img_list = sorted(glob.glob(os.path.join(full_frame_dir, "*.png")))
        
        # =========================================================
        # THE 2X SPEED MULTIPLIER: 
        # Throw away 50% of the paths BEFORE we read the images!
        # =========================================================
        img_list = img_list[::2]
        
        if len(img_list) == 0:
            raise FileNotFoundError(f"CRASH: No 256x256 frames found in {full_frame_dir}")

        label_list = []
        ann_list = []
        for phase in fi['label'].split(" "):
            if phase == '': continue
            if phase in self.dict.keys():
                val = self.dict[phase]
                label_list.append(val[0] if isinstance(val, (list, tuple)) else val)
                ann_list.append(phase)
                
        v_full, v_face, v_lh, v_rh = [], [], [], []

        def safe_read(path, fallback_shape):
            if not os.path.exists(path): return np.zeros(fallback_shape, dtype=np.uint8)
            img = cv2.imread(path)
            if img is None or img.size == 0: return np.zeros(fallback_shape, dtype=np.uint8)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Now the CPU only has to loop ~75 times instead of 150 times!
        for img_path in img_list:
            frame_name_png = os.path.basename(img_path)
            frame_name_jpg = frame_name_png.replace('.png', '.jpg')

            face_path = os.path.join(self.multicue_root, "face", self.mode, video_name, frame_name_jpg)
            lh_path = os.path.join(self.multicue_root, "left_hand", self.mode, video_name, frame_name_jpg)
            rh_path = os.path.join(self.multicue_root, "right_hand", self.mode, video_name, frame_name_jpg)

            v_full.append(safe_read(img_path, (256, 256, 3)))
            v_face.append(safe_read(face_path, (224, 224, 3)))
            v_lh.append(safe_read(lh_path, (224, 224, 3)))
            v_rh.append(safe_read(rh_path, (224, 224, 3)))

        v_full, label = self.full_aug(v_full, label_list)
        v_face, _ = self.crop_aug(v_face, label_list)
        v_lh, _ = self.crop_aug(v_lh, label_list)
        v_rh, _ = self.crop_aug(v_rh, label_list)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        def normalize_tensor(tensor_vid):
            tensor_vid = tensor_vid.float() / 255.0
            return (tensor_vid - mean) / std

        return normalize_tensor(v_full), normalize_tensor(v_face), normalize_tensor(v_lh), normalize_tensor(v_rh), label, ann_list, fi

    def transform_full(self):
        if self.transform_mode == "train":
            return video_augmentation.Compose([
                video_augmentation.RandomCrop(224),
                video_augmentation.ToTensor(),
            ])
        else:
            return video_augmentation.Compose([
                video_augmentation.CenterCrop(224),
                video_augmentation.ToTensor(),
            ])
            
    def transform_crops(self):
        return video_augmentation.Compose([
            video_augmentation.ToTensor(),
        ])

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        v_full, v_face, v_lh, v_rh, label, ann, info = list(zip(*batch))
        
        max_len = len(v_full[0])
        video_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 12 for vid in v_full])
        
        left_pad = 6
        right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 6
        total_max_len = max_len + left_pad + right_pad

        def pad_sequence(video_list):
            padded = []
            for vid in video_list:
                padded_vid = torch.cat((
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(total_max_len - len(vid) - left_pad, -1, -1, -1),
                ), dim=0)
                padded.append(padded_vid)
            return torch.stack(padded)

        pad_full = pad_sequence(v_full)
        pad_face = pad_sequence(v_face)
        pad_lh = pad_sequence(v_lh)
        pad_rh = pad_sequence(v_rh)

        label_length = torch.LongTensor([len(lab) for lab in label])
        if max(label_length) == 0:
            return pad_full, pad_face, pad_lh, pad_rh, video_length, [], [], info
        else:
            padded_label = []
            for lab in label: padded_label.extend(lab)
            return pad_full, pad_face, pad_lh, pad_rh, video_length, torch.LongTensor(padded_label), label_length, ann, info

    def __len__(self):
        return len(self.inputs_list)
