import copy
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.distributed as dist

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

        # --- PURE RESNET BACKBONE ---
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        # IAM and IEM
        self.cond1d_type = ["K5P2"]
        self.cond1d_size = ["5", "2"]

        # Input is 512 direct from ResNet
        self.conv1d = Conv1d(
            input_size=512,
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

    def extract_features(self, inputs, len_x):
        if len(inputs.shape) == 5:
            if inputs.shape[1] == 3:
                inputs = inputs.permute(0, 2, 1, 3, 4)
            b, t, c, h, w = inputs.shape
            inputs = inputs.reshape(b * t, c, h, w)

        res_feats = self.resnet(inputs)
        res_feats = res_feats.view(res_feats.size(0), -1)

        return res_feats

    def forward(self, x, len_x, label=None, label_lgt=None, ann=None):
        if len(x.shape) == 5:
            batch, temp, channel, height, width = x.shape
            inputs = x.reshape(batch * temp, channel, height, width)
            
            framewise = self.extract_features(inputs, len_x)
            framewise = framewise.view(batch, temp, 512).transpose(1, 2)
        else:
            framewise = x

        # 1. Visual Features (V)
        x = self.conv1d(framewise) 
        lgt = compute_lgt(len_x, self.cond1d_type, self.cond1d_size)

        # 2. Auxiliary Visual Prediction
        aux_logits = self.classifier(x)

        # 3. Contextual BiLSTM Prediction
        tm_outputs = self.temporal_model(x, lgt)  
        outputs = self.classifier(tm_outputs)

        pred = None if self.training \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)

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
            ve_weight = self.loss_weights['IteLoss']
            loss_dict['VELoss'] = ve_weight * self.loss['CTCLoss'](
                aux_logits.log_softmax(-1), label.cpu(), lgt.cpu(), label_lgt.cpu()
            )
            total_loss += loss_dict['VELoss']

        if 'ItaLoss' in self.loss_weights:
            va_weight = self.loss_weights['ItaLoss']
            tau = 8.0 

            student_log_probs = F.log_softmax(aux_logits / tau, dim=-1)
            teacher_probs = F.softmax(logits.detach() / tau, dim=-1)

            kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

            loss_dict['VALoss'] = va_weight * kl_loss * (tau ** 2)
            total_loss += loss_dict['VALoss']

        loss_dict['total_loss'] = total_loss
        return loss_dict

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
        return self.loss
