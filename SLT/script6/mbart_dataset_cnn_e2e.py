import os
import torch
import numpy as np
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer

class SemanticMBartDataset(Dataset):
    def __init__(self, stm_path, visual_dir, tokenizer, gloss_dict, max_length=128, max_vis_length=200):
        self.tokenizer = tokenizer
        self.tokenizer.src_lang = "de_DE"
        self.tokenizer.tgt_lang = "de_DE"
        self.max_length = max_length
        self.max_vis_length = max_vis_length 
        self.visual_dir = visual_dir
        self.gloss_dict = gloss_dict 

        print("1. Loading Translations and Groundtruth Glosses from STM...")
        
        targets_dict = {}
        gloss_targets_dict = {}
        
        with open(stm_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.split(',', 3)
                if len(parts) >= 4:
                    vid = parts[0].strip()
                    # Assuming Index 2 is Glosses and Index 3 is Translation
                    gloss_seq = parts[2].strip() 
                    translation = parts[3].strip()
                    
                    targets_dict[vid] = translation
                    gloss_targets_dict[vid] = gloss_seq

        # Find perfectly intersecting videos
        self.valid_video_ids = [
            vid for vid in targets_dict.keys()
            if os.path.exists(os.path.join(visual_dir, f"{vid}.npy"))
        ]

        self.target_texts = [targets_dict[vid] for vid in self.valid_video_ids]
        self.groundtruth_glosses = [gloss_targets_dict[vid] for vid in self.valid_video_ids]
        
        print(f"Found {len(self.valid_video_ids)} valid Triplets (Visual + Groundtruth Gloss + Translation).")

        print("2. Pre-computing perfect sBERT Semantic Vectors...")
        sbert_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.target_embeddings = sbert_model.encode(self.target_texts, convert_to_tensor=True).cpu()
        print("sBERT embeddings generated successfully.")

    def __len__(self):
        return len(self.valid_video_ids)

    def __getitem__(self, idx):
        vid = self.valid_video_ids[idx]
        target_translation = self.target_texts[idx]
        gt_gloss_seq = self.groundtruth_glosses[idx]

        # --- 1. Tokenize Placeholder Text for mBART ---
        model_inputs = self.tokenizer(
            gt_gloss_seq, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        labels = self.tokenizer(
            text_target=target_translation, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        input_ids = model_inputs["input_ids"].squeeze()
        attention_mask = model_inputs["attention_mask"].squeeze()
        labels_ids = labels["input_ids"].squeeze()
        labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100

        # --- 2. Load and Pad Visual Features ---
        npy_path = os.path.join(self.visual_dir, f"{vid}.npy")
        visual_feat = torch.tensor(np.load(npy_path), dtype=torch.float32) 
        
        T = visual_feat.shape[0]
        CNN_DIM = visual_feat.shape[1] 

        if T < self.max_vis_length:
            pad_size = self.max_vis_length - T
            pad_tensor = torch.zeros((pad_size, CNN_DIM), dtype=torch.float32)
            visual_feat = torch.cat([visual_feat, pad_tensor], dim=0)
            vis_mask = torch.cat([torch.ones(T), torch.zeros(pad_size)]) 
        else:
            visual_feat = visual_feat[:self.max_vis_length, :]
            vis_mask = torch.ones(self.max_vis_length)

        # --- 3. GENERATE PERFECT CTC TARGETS FOR BiLSTM ---
        gloss_ids = []
        for word in gt_gloss_seq.split():
            if word == '': continue
            if word in self.gloss_dict:
                val = self.gloss_dict[word]
                gloss_ids.append(val[0] if isinstance(val, (list, tuple)) else val)

        ctc_target_length = len(gloss_ids)

        # Pad to max_length with 0 (blank token)
        padded_ctc = gloss_ids + [0] * (self.max_length - len(gloss_ids))
        padded_ctc = padded_ctc[:self.max_length] 
        if ctc_target_length > self.max_length:
            ctc_target_length = self.max_length

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_ids,
            "target_embeds": self.target_embeddings[idx],
            "visual_features": visual_feat,
            "visual_mask": vis_mask,
            "ctc_gloss_targets": torch.tensor(padded_ctc, dtype=torch.long),
            "ctc_target_lengths": torch.tensor(ctc_target_length, dtype=torch.long)
        }
