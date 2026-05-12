import copy
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def unpad_padded(x, xl, dim=0):
    dims = list(range(len(x.shape)))
    dims.insert(0, dims.pop(dim))
    x = x.permute(*dims)
    return [xi[:xli] for xi, xli in zip(x, xl)]

def compute_lgt(lgt, kernel_type, kernel_size):
    feat_len = copy.deepcopy(lgt)
    for i in range(len(kernel_type)):
        feat_len -= int(kernel_size[0]) - 1
        feat_len = feat_len // 2
    return feat_len.cpu()

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=1, dp_ratio=0.3, bidirectional=True):
        super(BiLSTM, self).__init__()
        self.dp_ratio = dp_ratio
        self.num_layers = num_layers
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.hidden_size = int(hidden_size / 2) if self.bidirectional else hidden_size
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dp_ratio,
            bidirectional=self.bidirectional
        )

    def forward(self, x, lgt, hidden=None):
        packed_seq = nn.utils.rnn.pack_padded_sequence(x, lgt)
        out, hidden = self.lstm(packed_seq, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(out)
        return outputs

class Conv1d(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_type, kernel_size):
        super(Conv1d, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_type = kernel_type
        self.kernel_size = kernel_size
        self.temporal_conv = nn.ModuleList()
        for layer_idx in range(len(self.kernel_type)):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            self.temporal_conv.append(
                nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(self.kernel_size[0]), stride=1, padding=0)
            )
            self.temporal_conv.append(nn.BatchNorm1d(self.hidden_size))
            self.temporal_conv.append(nn.ReLU(inplace=True))
            self.temporal_conv.append(nn.MaxPool1d(kernel_size=int(self.kernel_size[1]), ceil_mode=False))

    def forward(self, vis_fea):
        for module in self.temporal_conv:
            vis_fea = module(vis_fea)
        return vis_fea.permute(2, 0, 1)

class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs

class SLRModel(nn.Module):
    def __init__(
            self, num_classes,
            hidden_size=512, gloss_dict=None, loss_weights=None,
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights

        # --- ASYMMETRIC BACKBONES (Alternative 1) ---
        # Full Stream: Heavy ResNet-18 (Output dim: 512)
        self.full_stream = nn.Sequential(*list(models.resnet18(weights=models.ResNet18_Weights.DEFAULT).children())[:-1])
        
        # Crop Streams: Lightweight MobileNet-V3-Small (Output dim: 576)
        self.face_stream = nn.Sequential(*list(models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).children())[:-1])
        self.lh_stream = nn.Sequential(*list(models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).children())[:-1])
        self.rh_stream = nn.Sequential(*list(models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).children())[:-1])

        # --- THE DEEP FREEZE ---
        # Freezes Conv1, BatchNorm, ReLU, MaxPool, Layer1, and Layer2
        # Leaves Layer3 and Layer4 trainable to learn hands/faces specifically.
        #def freeze_base_layers(model_stream):
        #    for i, child in enumerate(model_stream.children()):
        #        if i < 6: 
        #            for param in child.parameters():
        #                param.requires_grad = False
                        
        #freeze_base_layers(self.full_stream)
        #freeze_base_layers(self.face_stream)
        #freeze_base_layers(self.lh_stream)
        #freeze_base_layers(self.rh_stream)

        # --- THE FUSION BOTTLENECK ---
        # Total incoming dim: 512 (ResNet) + 576 (Face) + 576 (LH) + 576 (RH) = 2240
        self.fusion_layer = nn.Sequential(
            nn.Dropout(p=0.3),            # Penalize incoming ResNet vectors 
            nn.Linear(2240, hidden_size), # Compress 2240 -> 512
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(p=0.5)             # Heavy anti-memorization penalty
        )

        self.cond1d_type = ["K5P2"]
        self.cond1d_size = ["5", "2"]

        self.conv1d = Conv1d(
            input_size=hidden_size,
            hidden_size=hidden_size,
            kernel_type=self.cond1d_type,
            kernel_size=self.cond1d_size
        )

        self.decoder = utils.Decode(gloss_dict, num_classes)
        self.temporal_model = BiLSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True
        )
        self.classifier = NormLinear(hidden_size, self.num_classes)
        self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            if g is not None:
                g[g != g] = 0

    def extract_features(self, v_full, v_face, v_lh, v_rh):
        b, t, c, h, w = v_full.shape
        
        v_full = v_full.reshape(b * t, c, h, w)
        v_face = v_face.reshape(b * t, c, h, w)
        v_lh = v_lh.reshape(b * t, c, h, w)
        v_rh = v_rh.reshape(b * t, c, h, w)

        feat_full = self.full_stream(v_full).view(b * t, -1)
        feat_face = self.face_stream(v_face).view(b * t, -1)
        feat_lh = self.lh_stream(v_lh).view(b * t, -1)
        feat_rh = self.rh_stream(v_rh).view(b * t, -1)

        # --- AGGRESSIVE MODALITY DROPOUT (30%) ---
        if self.training:
            if torch.rand(1).item() < 0.20:
                feat_face = feat_face * 0.0
            
            if torch.rand(1).item() < 0.20:
                feat_lh = feat_lh * 0.0
                
            if torch.rand(1).item() < 0.20:
                feat_rh = feat_rh * 0.0

        multi_cue_tensor = torch.cat((feat_full, feat_face, feat_lh, feat_rh), dim=1)
        compressed_features = self.fusion_layer(multi_cue_tensor)

        return compressed_features

    def forward(self, vid_full, vid_face, vid_lh, vid_rh, len_x, label=None, label_lgt=None, ann=None):

        # ==========================================================
        # BLAZING FAST GPU AUGMENTATION BLOCK
        # ==========================================================
        if self.training:
            with torch.no_grad(): # Do not track gradients for augmentation!
                # GPU Color Jitter (Contrast and Brightness)
                c_factor = torch.empty(1).uniform_(0.8, 1.2).to(vid_full.device)
                bias = torch.empty(1).uniform_(-0.2, 0.2).to(vid_full.device)

                vid_full = vid_full * c_factor + bias
                vid_face = vid_face * c_factor + bias
                vid_lh = vid_lh * c_factor + bias
                vid_rh = vid_rh * c_factor + bias

                # GPU Tube Cutout
                def apply_tube_cutout(vid_tensor, max_box_size):
                    if torch.rand(1).item() < 0.5:
                        h, w = vid_tensor.shape[-2], vid_tensor.shape[-1]
                        box_h = torch.randint(16, max_box_size, (1,)).item()
                        box_w = torch.randint(16, max_box_size, (1,)).item()
                        y1 = torch.randint(0, h - box_h, (1,)).item()
                        x1 = torch.randint(0, w - box_w, (1,)).item()
                        # Set pixels to 0.0 (which is the ImageNet normalized mean)
                        vid_tensor[..., y1:y1+box_h, x1:x1+box_w] = 0.0 
                    return vid_tensor

                vid_full = apply_tube_cutout(vid_full, 64)
                vid_face = apply_tube_cutout(vid_face, 48)
                vid_lh = apply_tube_cutout(vid_lh, 48)
                vid_rh = apply_tube_cutout(vid_rh, 48)
        # ==========================================================

        batch, temp, channel, height, width = vid_full.shape
        
        framewise = self.extract_features(vid_full, vid_face, vid_lh, vid_rh)
        framewise = framewise.view(batch, temp, 512).transpose(1, 2)

        x = self.conv1d(framewise) 
        lgt = compute_lgt(len_x, self.cond1d_type, self.cond1d_size)

        aux_logits = self.classifier(x)
        tm_outputs = self.temporal_model(x, lgt)  
        outputs = self.classifier(tm_outputs)

        pred = None if self.training else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)

        return {
            "fea_lgt": lgt,
            "sequence_logits": outputs, 
            "aux_logits": aux_logits,   
            "recognized_sents": pred,
            "bilstm_features": tm_outputs,
            "conv1d_features": x
        }

    def losses_calculation(self, ret_dict, label, label_lgt):
        loss_dict = {}
        total_loss = 0.0
        logits = ret_dict['sequence_logits']
        aux_logits = ret_dict['aux_logits']
        lgt = ret_dict['fea_lgt']

        loss_dict['SeqCTC'] = self.loss_weights.get('SeqCTC', 1.0) * self.loss['CTCLoss'](
            logits.log_softmax(-1), label.cpu(), lgt.cpu(), label_lgt.cpu()
        )
        total_loss += loss_dict['SeqCTC']

        if 'IteLoss' in self.loss_weights:
            loss_dict['VELoss'] = self.loss_weights['IteLoss'] * self.loss['CTCLoss'](
                aux_logits.log_softmax(-1), label.cpu(), lgt.cpu(), label_lgt.cpu()
            )
            total_loss += loss_dict['VELoss']

        if 'ItaLoss' in self.loss_weights:
            tau = 8.0 
            student_log_probs = F.log_softmax(aux_logits / tau, dim=-1)
            teacher_probs = F.softmax(logits.detach() / tau, dim=-1)
            kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
            loss_dict['VALoss'] = self.loss_weights['ItaLoss'] * kl_loss * (tau ** 2)
            total_loss += loss_dict['VALoss']

        loss_dict['total_loss'] = total_loss
        return loss_dict

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
        return self.loss
