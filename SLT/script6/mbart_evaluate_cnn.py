import os
import json
import torch
import numpy as np
from tqdm import tqdm
import sacrebleu
from transformers import AutoTokenizer
from semantic_mbart_cnn import SemanticMBart  # Using your newly updated architecture file!
from sentence_transformers import SentenceTransformer, util
from peft import LoraConfig, get_peft_model

def main():
    # --- PATHS ---
    MODEL_ID = "facebook/mbart-large-50-many-to-many-mmt"

    # Pointing to your new CNN1D trained model
    WEIGHTS_PATH = "./work_dir/mBART_attention_cnn1d_mp/semantic_mbart_final.pt"
    
    # CRITICAL: Since you fixed the tokenizer bug in training, 
    # loading it from your cnn1d folder is now 100% safe and correct!
    TOKENIZER_PATH = "./work_dir/mBART_attention_cnn1d_mp/tokenizer"

    # NOTE: If you generated `dev_updated_glosses.json`, point this there instead!
    JSON_PATH = "./SLT/data/Phase1_gloss/dev_phase1_glosses.json"
    STM_PATH = "./SLT/data/phoenix_dev_final.stm"

    # CRITICAL: Point this to your DEV 1D CNN features!
    VISUAL_DIR = "./work_dir/Phase1_Resnet_mp/extracted_conv1d_features/dev"

    print("1. Loading Dev Dataset & Checking Visual Features...")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        inputs_dict = json.load(f)

    targets_dict = {}
    with open(STM_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split(',', 3)
            if len(parts) >= 4:
                targets_dict[parts[0].strip()] = parts[3].strip().lower()

    # Only evaluate videos that ALSO have an extracted .npy feature file
    valid_video_ids = [
        vid for vid in inputs_dict.keys()
        if vid in targets_dict and os.path.exists(os.path.join(VISUAL_DIR, f"{vid}.npy"))
    ]
    print(f"Loaded {len(valid_video_ids)} perfectly aligned CNN1D Multi-Modal examples.")

    print("2. Loading Tokenizer & Custom Model...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    tokenizer.src_lang = "de_DE"

    model = SemanticMBart(model_id=MODEL_ID, lambda_weight=1.0)
    model.mbart.resize_token_embeddings(len(tokenizer))

    print("Rebuilding EXACT LoRA architecture to match saved dictionary keys...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        modules_to_save=["model.shared", "lm_head"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )

    model.mbart = get_peft_model(model.mbart, lora_config)

    print("Loading custom trained CNN1D weights...")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True), strict=False)

    # =================================================================================
    # FATAL BUG FIX: Manually restore PEFT weight tying so Encoder/Decoder aren't blind!
    # Without this, mBART cannot see the German dictionary it just loaded!
    print("Restoring broken PEFT weight tying for Generation...")
    model.mbart.base_model.model.model.encoder.embed_tokens = model.mbart.base_model.model.model.shared
    model.mbart.base_model.model.model.decoder.embed_tokens = model.mbart.base_model.model.model.shared
    # =================================================================================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print("3. Loading sBERT for Semantic Grading...")
    sbert_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").to(device)

    references = []
    predictions = []

    print("4. Starting Batched Multi-Modal Evaluation...")
    forced_bos = tokenizer.lang_code_to_id["de_DE"]
    MAX_VIS_LENGTH = 200 # Must match the training script padding
    BATCH_SIZE = 8

    with torch.no_grad():
        for i in tqdm(range(0, len(valid_video_ids), BATCH_SIZE)):
            batch_vids = valid_video_ids[i : i + BATCH_SIZE]

            batch_source_texts = [inputs_dict[vid] for vid in batch_vids]
            batch_target_texts = [targets_dict[vid] for vid in batch_vids]

            # A. Tokenize Text (Padding required for batching!)
            inputs = tokenizer(
                batch_source_texts,
                return_tensors="pt",
                max_length=64,
                padding=True,
                truncation=True
            ).to(device)

            # B. Load and Pad 1D CNN Features
            batch_visual_feats = []
            batch_vis_masks = []

            for vid in batch_vids:
                npy_path = os.path.join(VISUAL_DIR, f"{vid}.npy")
                visual_feat = torch.tensor(np.load(npy_path), dtype=torch.float32).to(device)
                T = visual_feat.shape[0]
                CNN_DIM = visual_feat.shape[1]

                if T < MAX_VIS_LENGTH:
                    pad_size = MAX_VIS_LENGTH - T
                    pad_tensor = torch.zeros((pad_size, CNN_DIM), dtype=torch.float32).to(device)
                    visual_feat = torch.cat([visual_feat, pad_tensor], dim=0)
                    vis_mask = torch.cat([torch.ones(T), torch.zeros(pad_size)]).to(device)
                else:
                    visual_feat = visual_feat[:MAX_VIS_LENGTH, :]
                    vis_mask = torch.ones(MAX_VIS_LENGTH).to(device)

                batch_visual_feats.append(visual_feat)
                batch_vis_masks.append(vis_mask)

            # Shape becomes [Batch, 200, CNN_DIM]
            visual_feat = torch.stack(batch_visual_feats)
            vis_mask = torch.stack(batch_vis_masks)

            # C. --- MANUAL MULTI-MODAL FUSION (PURE ATTENTION UPGRADE) ---
            text_embeds = model.mbart.get_input_embeddings()(inputs["input_ids"])

            # Cast float32 features to match model precision (fp16) to prevent crashes
            visual_feat = visual_feat.to(text_embeds.dtype)

            # 1. Project 1D CNN features
            vis_embeds = model.visual_projection(visual_feat)

            # 2. Add Positional Embeddings (Time stamps)
            batch_size, seq_len, _ = vis_embeds.size()
            positions = torch.arange(seq_len, dtype=torch.long, device=device)
            positions = positions.unsqueeze(0).expand(batch_size, seq_len)
            vis_embeds = vis_embeds + model.pos_embedding(positions)

            # 3. Visual Transformer
            key_padding_mask = (vis_mask == 0).bool()
            vis_embeds = model.visual_transformer(
                vis_embeds,
                src_key_padding_mask=key_padding_mask
            )

            # 4. Cross Attention (Text looks at Time-Stamped Video)
            attn_output, _ = model.cross_attention(
                query=text_embeds,
                key=vis_embeds,
                value=vis_embeds,
                key_padding_mask=key_padding_mask
            )

            # 5. Final Fused Embeddings ready for generation
            fused_embeds = model.layer_norm(text_embeds + attn_output)

            # D. Generate using the FUSED embeddings (Bypassing input_ids)
            generated_tokens = model.mbart.generate(
                inputs_embeds=fused_embeds,
                attention_mask=inputs["attention_mask"],
                forced_bos_token_id=forced_bos,
                max_new_tokens=80,
                max_length=None,
                num_beams=5,
                length_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

            # E. Batch Decode
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            for pred_text, target_text in zip(decoded_preds, batch_target_texts):
                references.append(target_text)
                predictions.append(pred_text.lower())

    print("\n5. Calculating Final Scores...")
    bleu = sacrebleu.corpus_bleu(predictions, [references], lowercase=True)

    pred_embeds = sbert_model.encode(predictions, convert_to_tensor=True)
    target_embeds = sbert_model.encode(references, convert_to_tensor=True)
    cosine_scores = util.cos_sim(pred_embeds, target_embeds)
    semantic_acc = cosine_scores.diag().mean().item() * 100

    print("======================================")
    print("          FINAL GRADING REPORT        ")
    print("======================================")
    print(f"SacreBLEU Score:         {bleu.score:.2f}")
    print(f"sBERT Semantic Accuracy: {semantic_acc:.2f}%")
    print("======================================")

if __name__ == "__main__":
    main()
