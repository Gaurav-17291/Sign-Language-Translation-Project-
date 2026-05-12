import os
import json
import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from mbart_dataset_cnn import SemanticMBartDataset
from semantic_mbart_cnn import SemanticMBart

class SemanticTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs[0]
        return (loss, outputs) if return_outputs else loss

def main():
    MODEL_ID = "facebook/mbart-large-50-many-to-many-mmt"
    
    # CRITICAL: Using the updated glosses so mBART doesn't learn old mistakes!
    JSON_PATH = "./SLT/data/Phase1_gloss/train_phase1_glosses.json"
    STM_PATH = "./SLT/data/phoenix_train_final.stm"
    VISUAL_DIR = "./work_dir/Phase1_Resnet_mp/extracted_conv1d_features/train"
    
    OUTPUT_DIR = "./work_dir/mBART_attention_cnn1d_mp"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("1. Loading Default mBART Tokenizer (Independent from Phase 1)...")
    # THE FIX: Loading the pristine base tokenizer, not the one from mbart_train.py
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.src_lang = "de_DE"
    tokenizer.tgt_lang = "de_DE"

    print("2. Loading Dataset...")
    train_dataset = SemanticMBartDataset(JSON_PATH, STM_PATH, VISUAL_DIR, tokenizer, max_length=128, max_vis_length=200)

    print("3. Loading Custom Semantic mBART Model...")
    model = SemanticMBart(model_id=MODEL_ID, lambda_weight=1.0)

    print("4. Attaching LoRA Adapters...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    model.mbart = get_peft_model(model.mbart, lora_config)

    print(" -> Hard-Freezing Text Embeddings and LM Head to protect them and save VRAM...")
    for name, param in model.mbart.named_parameters():
        if "shared" in name or "embed_tokens" in name or "lm_head" in name:
            param.requires_grad = False

    # Show the user the REAL trainable parameter count
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\\nActual Trainable Params (including new visual layers): {trainable_params:,}\\n")

    print("5. Setting up Semantic Trainer...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=1e-4,
        logging_steps=20,
        num_train_epochs=10, 
        save_strategy="no",
        fp16=True,
        optim="adamw_bnb_8bit",
        report_to="none",
        remove_unused_columns=False
    )

    trainer = SemanticTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    print("IGNITION: Starting Multi-Modal Semantic Training!")
    trainer.train()

    print("Training Complete! Saving Architecture...")
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "semantic_mbart_final.pt"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "tokenizer"))
    print("Done!")

if __name__ == "__main__":
    main()
