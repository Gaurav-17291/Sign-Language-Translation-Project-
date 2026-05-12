import torch
import torch.nn as nn
import torch.nn.functional as F

def gen_gt(sentence):
    lgt = len(sentence)
    # FIXED: Use the same device as the input sentence instead of hardcoding .cuda()
    gt = -torch.ones(lgt, lgt, device=sentence.device)
    for i, gls in enumerate(sentence):
        for k in range(lgt):
            if sentence[k] == gls:
                gt[i, k] = 1.0
    return gt

class IteLoss(nn.Module):
    def __init__(self):
        super(IteLoss, self).__init__()

    def forward(self, gls_emd, vis_emd, label, label_lgt):
        # 1. Normalize features to calculate Cosine Similarity
        vis_emd = F.normalize(vis_emd, dim=-1)
        gls_emd = F.normalize(gls_emd, dim=-1)
        
        # 2. Matrix multiplication (Shape: [Batch, Time, Time])
        sim_matrix = torch.matmul(vis_emd, gls_emd.transpose(-1, -2)) 
        
        # 3. Safely calculate loss for Batch Size = 1
        valid_len = label_lgt[0].item()
        
        # Guard against empty sequences
        if valid_len == 0:
            return torch.tensor(0.0, device=vis_emd.device, requires_grad=True)
            
        gt_matrix = gen_gt(label[0][:valid_len])
        
        # 4. Crop similarity matrix to valid sequence length and calculate MSE
        valid_sim = sim_matrix[0, :valid_len, :valid_len]
        loss = F.mse_loss(valid_sim, gt_matrix)
        
        return loss

class ItaLoss(nn.Module):
    def __init__(self):
        super(ItaLoss, self).__init__()

    def forward(self, gls_emd, vis_emd, label, label_lgt):
        # Image-Text Alignment: Bringing visual and text embeddings directly together
        valid_len = label_lgt[0].item()
        
        # Guard against empty sequences
        if valid_len == 0:
            return torch.tensor(0.0, device=vis_emd.device, requires_grad=True)
            
        # Crop to the actual length of the sign language sequence
        v_valid = vis_emd[0, :valid_len, :]
        g_valid = gls_emd[0, :valid_len, :]
        
        # Calculate alignment loss
        loss = F.mse_loss(v_valid, g_valid)
        return loss
