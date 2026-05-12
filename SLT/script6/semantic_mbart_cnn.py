import torch
import torch.nn as nn
from transformers import MBartForConditionalGeneration

class SemanticMBart(nn.Module):
    def __init__(self, model_id="facebook/mbart-large-50-many-to-many-mmt", lambda_weight=1.0):
        super().__init__()
        self.lambda_weight = lambda_weight

        self.mbart = MBartForConditionalGeneration.from_pretrained(
            model_id,
            use_safetensors=True
        )
        d_model = self.mbart.config.d_model # 1024

        self.projection_head = nn.Linear(d_model, 384)
        self.mse_loss = nn.MSELoss()

        # --- NEW MULTI-MODAL LAYERS (TRANSFORMER UPGRADE) ---
        # 1. Project 1D CNN features (512) to mBART dims (1024)
        self.visual_projection = nn.Linear(512, d_model)

        # 2. FIXED: Positional Embedding increased to 300 to handle max_vis_length=200
        self.pos_embedding = nn.Embedding(300, d_model)

        # 3. The Transformer Adapter (2 Layers of Self-Attention)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=2048,
            dropout=0.1, activation="gelu", batch_first=True
        )
        self.visual_transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)

        # 4. Cross-Attention
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)

    def gradient_checkpointing_enable(self, **kwargs):
        """Catches the Trainer's checkpoint command and passes it to the inner mBART model."""
        self.mbart.gradient_checkpointing_enable(**kwargs)

    def forward(self, input_ids, attention_mask, labels=None, target_embeds=None, visual_features=None, visual_mask=None):
        inputs_embeds = self.mbart.get_input_embeddings()(input_ids)

        if visual_features is not None:
            # Cast the visual features to match the model's mixed precision (fp16)
            visual_features = visual_features.to(inputs_embeds.dtype)

            # 1. Project visual features to 1024
            vis_embeds = self.visual_projection(visual_features)

            # 2. ADD TIME: Generate positions and add Positional Embeddings
            batch_size, seq_len, _ = vis_embeds.size()
            positions = torch.arange(seq_len, dtype=torch.long, device=vis_embeds.device)
            positions = positions.unsqueeze(0).expand(batch_size, seq_len)
            vis_embeds = vis_embeds + self.pos_embedding(positions)

            # 3. PyTorch Attention Padding Mask
            key_padding_mask = (visual_mask == 0).bool() if visual_mask is not None else None

            # 4. Apply Visual Self-Attention
            vis_embeds = self.visual_transformer(vis_embeds, src_key_padding_mask=key_padding_mask)

            # 5. Cross Attention
            attn_output, _ = self.cross_attention(
                query=inputs_embeds, key=vis_embeds, value=vis_embeds, key_padding_mask=key_padding_mask
            )

            # 6. Add & Normalize
            inputs_embeds = self.layer_norm(inputs_embeds + attn_output)

        # C. Pass the FUSED embeddings into mBART
        outputs = self.mbart(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )

        ce_loss = outputs.loss
        total_loss = ce_loss

        # D. Calculate Semantic Loss
        if labels is not None and target_embeds is not None:
            encoder_states = outputs.encoder_last_hidden_state

            # FIX 1: Safely cast mask to match encoder's exact dtype (prevents fp16/fp32 multiplication crashes)
            mask_expanded = attention_mask.unsqueeze(-1).expand(encoder_states.size()).to(encoder_states.dtype)
            
            sum_embeddings = torch.sum(encoder_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_encoder = sum_embeddings / sum_mask

            projected_encoder = self.projection_head(pooled_encoder)
            
            # FIX 2: Safely cast target dataloader features to match the model's fp16 output
            target_embeds = target_embeds.to(projected_encoder.dtype)
            mse = self.mse_loss(projected_encoder, target_embeds)

            total_loss = ce_loss + (self.lambda_weight * mse)

        return (total_loss, outputs.logits)
