import os
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from mbart_dataset_cnn_e2e import SemanticMBartDataset
from semantic_mbart_cnn_e2e import SemanticMBart # Updated import name

class DynamicSemanticTrainer(Trainer):
    def __init__(self, mbart_tokenizer, gloss_vocab, blank_id=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mbart_tokenizer = mbart_tokenizer
        self.gloss_vocab = gloss_vocab 
        self.blank_id = blank_id
        
        self.ctc_loss = nn.CTCLoss(blank=self.blank_id, zero_infinity=True)

    def greedy_ctc_decode(self, logits):
        pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
        decoded_glosses = []
        for i in range(len(pred_ids)):
            if pred_ids[i] != self.blank_id and (i == 0 or pred_ids[i] != pred_ids[i-1]):
                decoded_glosses.append(self.gloss_vocab.get(pred_ids[i], "<UNK>"))
        return " ".join(decoded_glosses).strip()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        visual_feat = inputs["visual_features"]
        vis_mask = inputs["visual_mask"]

        # THE FIX: Call the DDP wrapper directly and pass the toggle switch!
        # Do NOT unwrap the model. DDP handles the sync safely this way.
        gloss_logits, input_lengths = model(
            visual_features=visual_feat, 
            visual_mask=vis_mask, 
            return_gloss_only=True
        )
        ctc_logits = gloss_logits.transpose(0, 1).log_softmax(2)
        
        # --- ACTIVATED CTC LOSS (FIX: Using .pop() to prevent kwargs crash!) ---
        ctc_targets = inputs.pop("ctc_gloss_targets", None)
        target_lengths = inputs.pop("ctc_target_lengths", None)
        
        if ctc_targets is not None and target_lengths is not None:
            loss_ctc = self.ctc_loss(ctc_logits, ctc_targets, input_lengths, target_lengths)
        else:
            loss_ctc = 0.0 # Safety fallback
            print("WARNING: ctc_gloss_targets not found in dataloader inputs!")

        with torch.no_grad():
            dynamic_strings = [self.greedy_ctc_decode(logits) for logits in gloss_logits]

        new_text_inputs = self.mbart_tokenizer(
            dynamic_strings,
            return_tensors="pt",
            max_length=64, 
            padding="max_length",
            truncation=True
        ).to(visual_feat.device)

        inputs["input_ids"] = new_text_inputs["input_ids"]
        inputs["attention_mask"] = new_text_inputs["attention_mask"]

        # Now `inputs` no longer contains the CTC keys, so this will run perfectly
        outputs = model(**inputs)
        loss_translation = outputs[0]
        
        # CTC Loss is often large. You can tune this 0.5 multiplier if one model branch dominates the other.
        ctc_alpha = 0.5 
        total_loss = loss_translation + (ctc_alpha * loss_ctc)
        
        return (total_loss, outputs) if return_outputs else total_loss

def main():
    MODEL_ID = "facebook/mbart-large-50-many-to-many-mmt"
    
    # --- CONSOLIDATED PATHS ---
    STM_PATH = "./SLT/data/phoenix_train_final.stm"
    VISUAL_DIR = "./work_dir/Phase1_Resnet_mp/extracted_conv1d_features/train"
    # --------------------------
    
    # --- DYNAMIC DICTIONARY LOADING ---
    DICT_PATH = "./datainfo/gloss_dict.npy"
    print(f"1. Loading Phase 1 Dictionary from {DICT_PATH}...")
    gloss_dict = np.load(DICT_PATH, allow_pickle=True).item()
    
    # Reverse dict for decoding (ID -> String)
    gloss_vocab = {v: k for k, v in gloss_dict.items()}
    gloss_vocab[0] = "<blank>"
    
    NUM_GLOSSES = len(gloss_dict) + 1 
    print(f" -> Found {len(gloss_dict)} glosses. Total Vocab Size: {NUM_GLOSSES}")

    OUTPUT_DIR = "./work_dir/mBART_attention_dynamic_mp"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("2. Loading PRE-TRAINED Phase 2 Tokenizer...")
    TOKENIZER_DIR = "./work_dir/mBART_attention_cnn1d_mp/tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    
    print("3. Loading Dataset...")
    train_dataset = SemanticMBartDataset(
        stm_path=STM_PATH, 
        visual_dir=VISUAL_DIR, 
        tokenizer=tokenizer, 
        gloss_dict=gloss_dict, 
        max_length=128, 
        max_vis_length=200
    )

    print("4. Loading Y-Split Semantic mBART Model...")
    model = SemanticMBart(model_id=MODEL_ID, lambda_weight=1.0, num_glosses=NUM_GLOSSES)
    model.mbart.resize_token_embeddings(len(tokenizer))

    print("5. Attaching LoRA Adapters...")
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        modules_to_save=["model.shared", "lm_head"],
        lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM"
    )
    model.mbart = get_peft_model(model.mbart, lora_config)

    print("6. Stitching Brains Together...")
    # CRITICAL FIX 1: Point to the model that just scored 21 BLEU!
    PHASE2_WEIGHTS = "./work_dir/mBART_attention_cnn1d_mp/semantic_mbart_final.pt"
    if os.path.exists(PHASE2_WEIGHTS):
        phase2_dict = torch.load(PHASE2_WEIGHTS, map_location="cpu", weights_only=True)
        
        # CRITICAL FIX 2: NO FILTERING! Load the entire trained visual pipeline AND text brain!
        model.load_state_dict(phase2_dict, strict=False)
        print(" -> Phase 2 CNN1D Visual Pipeline and Text Brain loaded successfully!")

    print(" -> Restoring broken PEFT weight tying...")
    model.mbart.base_model.model.model.encoder.embed_tokens = model.mbart.base_model.model.model.shared
    model.mbart.base_model.model.model.decoder.embed_tokens = model.mbart.base_model.model.model.shared

    print(" -> Freezing Text Embeddings and LM Head...")
    for name, param in model.mbart.named_parameters():
        if "modules_to_save" in name:
            param.requires_grad = False

    # --- PERFECT GRAFTING ---
    BILSTM_WEIGHTS_PATH = "./work_dir/Phase1_Resnet_mp/_best_model.pt" 
    print(f" -> Loading Phase 1 BiLSTM & Classifier Weights from {BILSTM_WEIGHTS_PATH}...")
    phase1_checkpoint = torch.load(BILSTM_WEIGHTS_PATH, map_location="cpu",weights_only=False)
    phase1_state_dict = phase1_checkpoint['model_state_dict']
    
    bilstm_weights = {k.replace('temporal_model.', ''): v for k, v in phase1_state_dict.items() if 'temporal_model' in k}
    classifier_weights = {k.replace('classifier.', ''): v for k, v in phase1_state_dict.items() if 'classifier' in k}
    
    model.temporal_model.load_state_dict(bilstm_weights, strict=True)
    model.gloss_classifier.load_state_dict(classifier_weights, strict=True)
    print(" -> Phase 1 BiLSTM & NormLinear Classifier perfectly grafted!")

    print("7. Setting up DYNAMIC Semantic Trainer...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=1e-5, 
        logging_steps=20,
        num_train_epochs=10, 
        save_strategy="no",
        fp16=True,
        optim="adamw_bnb_8bit",
        report_to="none",
        remove_unused_columns=False,
        # THE FIX: This stops DDP from incorrectly sweeping the branches and crashing!
        ddp_find_unused_parameters=False
    )

    trainer = DynamicSemanticTrainer(
        mbart_tokenizer=tokenizer,
        gloss_vocab=gloss_vocab,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    print("IGNITION: Starting DYNAMIC Y-Split Training!")
    trainer.train()

    print("Training Complete! Saving Architecture...")
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "semantic_mbart_final.pt"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "tokenizer"))
    print("Done!")

if __name__ == "__main__":
    main()
