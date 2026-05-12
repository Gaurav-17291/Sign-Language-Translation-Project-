import os
import torch
import numpy as np
from tqdm import tqdm
import sacrebleu
from transformers import AutoTokenizer
from semantic_mbart_cnn_e2e import SemanticMBart  # Using the E2E Y-Split Architecture!
from sentence_transformers import SentenceTransformer, util
from peft import LoraConfig, get_peft_model

def main():
    # --- PATHS ---
    MODEL_ID = "facebook/mbart-large-50-many-to-many-mmt"

    # Point to the new DYNAMIC training output directory
    WEIGHTS_PATH = "./work_dir/mBART_attention_dynamic_mp/semantic_mbart_final.pt"
    TOKENIZER_PATH = "./work_dir/mBART_attention_dynamic_mp/tokenizer"

    # Evaluation Data Paths
    STM_PATH = "./SLT/data/phoenix_dev_final.stm"
    VISUAL_DIR = "./work_dir/Phase1_Resnet_mp/extracted_conv1d_features/dev"
    DICT_PATH = "./datainfo/gloss_dict.npy"

    print("1. Loading Phase 1 Dictionary...")
    gloss_dict = np.load(DICT_PATH, allow_pickle=True).item()
    gloss_vocab = {v: k for k, v in gloss_dict.items()}
    gloss_vocab[0] = "<blank>"
    NUM_GLOSSES = len(gloss_dict) + 1 

    print("2. Loading Dev Dataset & Checking Visual Features...")
    targets_dict = {}
    with open(STM_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split(',', 3)
            if len(parts) >= 4:
                targets_dict[parts[0].strip()] = parts[3].strip().lower()

    # Find videos that have visual features
    valid_video_ids = [
        vid for vid in targets_dict.keys()
        if os.path.exists(os.path.join(VISUAL_DIR, f"{vid}.npy"))
    ]
    print(f"Loaded {len(valid_video_ids)} aligned CNN1D Multi-Modal examples.")

    print("3. Loading Tokenizer & E2E Custom Model...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    tokenizer.src_lang = "de_DE"

    # Initialize with the correct number of glosses for the BiLSTM
    model = SemanticMBart(model_id=MODEL_ID, lambda_weight=1.0, num_glosses=NUM_GLOSSES)
    model.mbart.resize_token_embeddings(len(tokenizer))

    print("Rebuilding EXACT LoRA architecture...")
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

    print(f"Loading custom trained DYNAMIC weights from {WEIGHTS_PATH}...")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True), strict=False)

    # Restore broken PEFT weight tying for Generation
    print("Restoring broken PEFT weight tying for Generation...")
    model.mbart.base_model.model.model.encoder.embed_tokens = model.mbart.base_model.model.model.shared
    model.mbart.base_model.model.model.decoder.embed_tokens = model.mbart.base_model.model.model.shared

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print("4. Loading sBERT for Semantic Grading...")
    sbert_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").to(device)

    references = []
    predictions = []

    print("5. Starting Batched Multi-Modal Evaluation with ON-THE-FLY Decoding...")
    forced_bos = tokenizer.lang_code_to_id["de_DE"]
    MAX_VIS_LENGTH = 200 
    BATCH_SIZE = 8

    with torch.no_grad():
        for i in tqdm(range(0, len(valid_video_ids), BATCH_SIZE)):
            batch_vids = valid_video_ids[i : i + BATCH_SIZE]
            batch_target_texts = [targets_dict[vid] for vid in batch_vids]

            # A. Load and Pad 1D CNN Features
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

            visual_feat = torch.stack(batch_visual_feats)
            vis_mask = torch.stack(batch_vis_masks)

            # B. --- ON-THE-FLY GLOSS GENERATION (BiLSTM) ---
            gloss_logits, _ = model(
                visual_features=visual_feat, 
                visual_mask=vis_mask, 
                return_gloss_only=True
            )
            
            # Greedy CTC Decode
            pred_ids = torch.argmax(gloss_logits, dim=-1).cpu().numpy()
            dynamic_strings = []
            for b in range(len(pred_ids)):
                decoded_glosses = []
                for j in range(len(pred_ids[b])):
                    if pred_ids[b][j] != 0 and (j == 0 or pred_ids[b][j] != pred_ids[b][j-1]):
                        decoded_glosses.append(gloss_vocab.get(pred_ids[b][j], ""))
                dynamic_strings.append(" ".join(decoded_glosses).strip())

            # C. Tokenize the freshly generated dynamic strings
            inputs = tokenizer(
                dynamic_strings,
                return_tensors="pt",
                max_length=64,
                padding=True,
                truncation=True
            ).to(device)

            # D. --- MANUAL MULTI-MODAL FUSION ---
            text_embeds = model.mbart.get_input_embeddings()(inputs["input_ids"])
            visual_feat_fp16 = visual_feat.to(text_embeds.dtype)

            # Project and Add Time
            vis_embeds = model.visual_projection(visual_feat_fp16)
            batch_size, seq_len, _ = vis_embeds.size()
            positions = torch.arange(seq_len, dtype=torch.long, device=device)
            positions = positions.unsqueeze(0).expand(batch_size, seq_len)
            vis_embeds = vis_embeds + model.pos_embedding(positions)

            # Transform and Cross-Attend
            key_padding_mask = (vis_mask == 0).bool()
            vis_embeds = model.visual_transformer(vis_embeds, src_key_padding_mask=key_padding_mask)
            attn_output, _ = model.cross_attention(
                query=text_embeds,
                key=vis_embeds,
                value=vis_embeds,
                key_padding_mask=key_padding_mask
            )
            fused_embeds = model.layer_norm(text_embeds + attn_output)

            # E. Generate Translations
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

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            for pred_text, target_text in zip(decoded_preds, batch_target_texts):
                references.append(target_text)
                predictions.append(pred_text.lower())

    print("\n6. Calculating Final Scores...")
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
