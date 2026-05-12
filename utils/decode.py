import torch
from itertools import groupby

class Decode(object):
    def __init__(self, gloss_dict, num_classes, search_mode="max", blank_id=0):
        self.i2g_dict = dict((v[0] if isinstance(v, (list, tuple)) else v, k) for k, v in gloss_dict.items())
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id
        
        if self.search_mode == "beam":
            import ctcdecode
            vocab = [chr(x) for x in range(20000, 20000 + num_classes)]
            self.ctc_decoder = ctcdecode.CTCBeamDecoder(
                vocab, beam_width=10, blank_id=blank_id, num_processes=10
            )

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
            
        if self.search_mode == "max":
            return self.MaxDecode(nn_output, vid_lgt)
        else:
            return self.BeamSearch(nn_output, vid_lgt, probs)

    def MaxDecode(self, nn_output, vid_lgt):
        # the class ID with the highest probability at each frame
        _, max_idx = torch.max(nn_output, dim=-1)
        max_idx = max_idx.cpu().numpy()
        vid_lgt = vid_lgt.cpu().numpy()
        
        ret_list = []
        for batch_idx in range(len(nn_output)):
            # Slice to the valid length of the sequence
            seq = max_idx[batch_idx][:vid_lgt[batch_idx]]
            
            # Collapse consecutive repeated predictions
            collapsed = [x[0] for x in groupby(seq)]
            
            # Removing CTC blank tokens
            final_seq = [x for x in collapsed if x != self.blank_id]
            
            # Map IDs back to gloss strings
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in enumerate(final_seq)])
            
        return ret_list

    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        if not probs:
            nn_output = nn_output.softmax(-1).cpu()
        vid_lgt = vid_lgt.cpu()
        beam_result, beam_scores, timesteps, out_seq_len = self.ctc_decoder.decode(nn_output, vid_lgt)
        
        ret_list = []
        for batch_idx in range(len(nn_output)):
            first_result = beam_result[batch_idx][0][:out_seq_len[batch_idx][0]]
            if len(first_result) != 0:
                first_result = torch.stack([x[0] for x in groupby(first_result)])
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in enumerate(first_result)])
            
        return ret_list
