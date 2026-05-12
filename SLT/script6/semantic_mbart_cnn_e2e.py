import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MBartForConditionalGeneration

# --- IMPORTED EXACTLY FROM PHASE 1 ---
class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs

class BiLSTM(nn.Module):
    def __init__(self, input_size=512, hidden_size=512, num_layers=2, dp_ratio=0.3, bidirectional=True):
        super(BiLSTM, self).__init__()
        self.dp_ratio = dp_ratio
        self.num_layers = num_layers
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.hidden_size = int(hidden_size / 2) if self.bidirectional else hidden_size
        self.lstm = nn.LSTM(
            input_size=self.input_size, hidden_size=self.hidden_size,
            num_layers=self.num_layers, dropout=self.dp_ratio, bidirectional=self.bidirectional
        )

    def forward(self, x, lgt, hidden=None):
        packed_seq = nn.utils.rnn.pack_padded_sequence(x, lgt, enforce_sorted=False)
        out, hidden = self.lstm(packed_seq, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(out)
        return outputs

class SemanticMBart(nn.Module):
    def __init__(self, model_id="facebook/mbart-large-50-many-to-many-mmt", lambda_weight=1.0, num_glosses=1296):
        super().__init__()
        self.lambda_weight = lambda_weight
        
        self.mbart = MBartForConditionalGeneration.from_pretrained(model_id, use_safetensors=True)
        d_model = self.mbart.config.d_model 
        
        self.projection_head = nn.Linear(d_model, 384)
        self.mse_loss = nn.MSELoss()

        # --- BRANCH 1: THE ALIGNMENT MODULE (BiLSTM + CTC) ---
        self.temporal_model = BiLSTM(input_size=512, hidden_size=512, num_layers=2, bidirectional=True)
        # THE FIX: Using Phase 1's custom NormLinear
        self.gloss_classifier = NormLinear(512, num_glosses)

        # --- BRANCH 2: THE PURE ATTENTION TRANSLATION MODULE ---
        self.visual_projection = nn.Linear(512, d_model)
        self.pos_embedding = nn.Embedding(300, d_model)
        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=2048, 
            dropout=0.1, activation="gelu", batch_first=True
        )
        self.visual_transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)

    def gradient_checkpointing_enable(self, **kwargs):
        self.mbart.gradient_checkpointing_enable(**kwargs)

    def forward_gloss(self, visual_features, visual_mask):
        visual_features = visual_features.to(torch.float32) 
        vis_transposed = visual_features.transpose(0, 1)
        lgt = visual_mask.sum(dim=1).cpu().long()
        
        bilstm_out = self.temporal_model(vis_transposed, lgt) 
        bilstm_out = bilstm_out.transpose(0, 1) 
        
        logits = self.gloss_classifier(bilstm_out)
        return logits, lgt

    # THE FIX: Added return_gloss_only=False as a toggle switch
    def forward(self, input_ids=None, attention_mask=None, labels=None, target_embeds=None, visual_features=None, visual_mask=None, return_gloss_only=False):
        
        # If the Trainer asks for glosses, route it to the BiLSTM and return immediately!
        if return_gloss_only:
            return self.forward_gloss(visual_features, visual_mask)

        # ... (The rest of your normal translation forward pass stays exactly the same) ...
        inputs_embeds = self.mbart.get_input_embeddings()(input_ids)

        if visual_features is not None:
            visual_features = visual_features.to(inputs_embeds.dtype)
            vis_embeds = self.visual_projection(visual_features)
            
            batch_size, seq_len, _ = vis_embeds.size()
            positions = torch.arange(seq_len, dtype=torch.long, device=vis_embeds.device)
            positions = positions.unsqueeze(0).expand(batch_size, seq_len)
            vis_embeds = vis_embeds + self.pos_embedding(positions)
            
            key_padding_mask = (visual_mask == 0).bool() if visual_mask is not None else None
            
            vis_embeds = self.visual_transformer(vis_embeds, src_key_padding_mask=key_padding_mask)
            
            attn_output, _ = self.cross_attention(
                query=inputs_embeds, key=vis_embeds, value=vis_embeds, key_padding_mask=key_padding_mask
            )
            inputs_embeds = self.layer_norm(inputs_embeds + attn_output)

        outputs = self.mbart(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            labels=labels, output_hidden_states=True
        )
        
        ce_loss = outputs.loss 
        total_loss = ce_loss
        
        if labels is not None and target_embeds is not None:
            encoder_states = outputs.encoder_last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(encoder_states.size()).to(encoder_states.dtype)
            sum_embeddings = torch.sum(encoder_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_encoder = sum_embeddings / sum_mask
            
            projected_encoder = self.projection_head(pooled_encoder)
            target_embeds = target_embeds.to(projected_encoder.dtype)
            mse = self.mse_loss(projected_encoder, target_embeds)
            
            total_loss = ce_loss + (self.lambda_weight * mse)

        return (total_loss, outputs.logits)
