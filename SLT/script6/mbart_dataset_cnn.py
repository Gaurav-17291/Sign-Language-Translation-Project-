import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer

class SemanticMBartDataset(Dataset):
    def __init__(self, json_path, stm_path, visual_dir, tokenizer, max_length=128, max_vis_length=300):
        self.tokenizer = tokenizer
        self.tokenizer.src_lang = "de_DE"
        self.tokenizer.tgt_lang = "de_DE"
        self.max_length = max_length
        self.max_vis_length = max_vis_length # Added to handle variable video lengths
        self.visual_dir = visual_dir

        print("1. Loading JSON and STM files...")
        with open(json_path, 'r', encoding='utf-8') as f:
            inputs_dict = json.load(f)

        targets_dict = {}
        with open(stm_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.split(',', 3)
                if len(parts) >= 4:
                    targets_dict[parts[0].strip()] = parts[3].strip()

        # Find intersecting videos that ALSO have an .npy visual feature file
        self.valid_video_ids = [
            vid for vid in inputs_dict.keys()
            if vid in targets_dict and os.path.exists(os.path.join(visual_dir, f"{vid}.npy"))
        ]

        self.source_texts = [inputs_dict[vid] for vid in self.valid_video_ids]
        self.target_texts = [targets_dict[vid] for vid in self.valid_video_ids]
        print(f"Found {len(self.valid_video_ids)} valid Triplets (Text + Visual + Target).")

        print("2. Pre-computing perfect sBERT Semantic Vectors...")
        sbert_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.target_embeddings = sbert_model.encode(self.target_texts, convert_to_tensor=True).cpu()
        print("sBERT embeddings generated successfully.")

    def __len__(self):
        return len(self.valid_video_ids)

    def __getitem__(self, idx):
        vid = self.valid_video_ids[idx]
        source = self.source_texts[idx]
        target = self.target_texts[idx]

        # 1. Tokenize Text
        model_inputs = self.tokenizer(
            source, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        labels = self.tokenizer(
            text_target=target, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        input_ids = model_inputs["input_ids"].squeeze()
        attention_mask = model_inputs["attention_mask"].squeeze()
        labels_ids = labels["input_ids"].squeeze()
        labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100

        # 2. Load and Pad Visual Features
        npy_path = os.path.join(self.visual_dir, f"{vid}.npy")
        visual_feat = torch.tensor(np.load(npy_path), dtype=torch.float32) 
        
        # Make the dimension dynamic!
        T = visual_feat.shape[0]
        CNN_DIM = visual_feat.shape[1] 

        # Pad to max_vis_length so batches have uniform shapes
        if T < self.max_vis_length:
            pad_size = self.max_vis_length - T
            # Use CNN_DIM instead of hardcoded 512
            pad_tensor = torch.zeros((pad_size, CNN_DIM), dtype=torch.float32)
            visual_feat = torch.cat([visual_feat, pad_tensor], dim=0)
            vis_mask = torch.cat([torch.ones(T), torch.zeros(pad_size)]) 
        else:
            visual_feat = visual_feat[:self.max_vis_length, :]
            vis_mask = torch.ones(self.max_vis_length)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_ids,
            "target_embeds": self.target_embeddings[idx],
            "visual_features": visual_feat,
            "visual_mask": vis_mask
        }
