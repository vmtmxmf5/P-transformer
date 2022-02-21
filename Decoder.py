import torch
import torch.nn as nn

from MHA import MultiHeadedAttention
from FeedForward import PositionwiseFeedForward
from PositionalEncoding import *


# pad mask
def sequence_mask(lengths, max_len=None):
    # 주의! pe에 max_len과 다름! Batch내 seq_len 중 가장 긴 길이를 의미!
    # lengths : LongTensor([len(sample_1), len(sample_2), ..., len(sample_n)]) (B,)
    batch_size = lengths.numel() # tensor element 개수
    max_len = max_len or lengths.max() # tensor(len(sample_i))
    # first_arg.lt(second_arg), 두번째 arg 미만이면 True (brodcastable)
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)            # 꼭 필요하지는 않음
            .repeat(batch_size, 1)       # (B, max_len)
            .lt(lengths.unsqueeze(1)))   # len(sample_i) 미만은 True, 이외에는 False


class DecoderLayer(nn.Module):
    def __init__(self,
                d_model,
                nhead,
                d_ff,
                dropout,
                attention_dropout,
                self_attn_type='scaled-dot'):
        super().__init__()

        if self_attn_type == 'scaled-dot':
            self.self_attn = MultiHeadedAttention(nhead,
                                                  d_model,
                                                  dropout=attention_dropout)
        self.context_attn = MultiHeadedAttention(nhead,
                                                 d_model,
                                                 dropout=attention_dropout)
        self.feed_forward = PositionwiseFeedForward(d_model,
                                                    d_ff,
                                                    dropout)
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, inputs, memory, src_pad_mask, tgt_pad_mask, step=None, future=False):
        # inputs :       (B, tgt_len, d_model)  // FloatTensor
        # memory :  (B, src_len, d_model)  // FloatTensor
        # src_pad_mask : (B, 1, src_len)        // bool
        # tgt_pad_mask : (B, 1, tgt_len)        // bool
        # step :         int
        # future == True ==> future_mask 사용 X  

        # output :       (B, tgt_len, d_model)
        # attns :        (B, nhead, tgt_len, src_len)
        
        ## 1) Mask (Training only)
        dec_mask = None
        if inputs.size(1) > 1:
            ## pad, upper triangle 만 True
            dec_mask = self._compute_dec_mask(tgt_pad_mask, future)
        
        ## 2) Pre-LN + self attn
        inputs_norm = self.layer_norm1(inputs)
        query, _ = self._forward_self_attn(inputs_norm, dec_mask, step)
        query = self.drop(query) + inputs
        
        ## 3) Pre-LN + context attn
        query_norm = self.layer_norm2(query)
        mid, attns = self.context_attn(query_norm,
                                       memory,
                                       memory,
                                       mask=src_pad_mask,
                                       attn_type="context")
        output = self.feed_forward(self.drop(mid) + query)
        return output, attns
        
    def _compute_dec_mask(self, tgt_pad_mask, future):
        # tgt_pad_mask : (B, 1, tgt_len) // bool
        ## pad_idx만 True
        tgt_len = tgt_pad_mask.size(-1)
        if not future:
            # future_mask : (tgt_len, tgt_len)
            future_mask = torch.ones([tgt_len, tgt_len],
                                    device=tgt_pad_mask.device,
                                    dtype=torch.uint8)
            future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len) # (1, tgt_len, tgt_len)

            try:
                ## upper triangle만 True
                future_mask = future_mask.bool()
            except AttributeError:
                pass
            ## torch.gt(A, 0) : A elements > 0 이면 True
            ## pad와 upper triangle은 True, 그 이외에는 False
            dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)
        else:
            dec_mask = tgt_pad_mask
        return dec_mask
    
    def _forward_self_attn(self, inputs_norm, dec_mask, step):
        if isinstance(self.self_attn, MultiHeadedAttention):
            return self.self_attn(inputs_norm,
                                inputs_norm,
                                inputs_norm,
                                mask=dec_mask,
                                attn_type='self')

        
class Decoder(nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 nhead,
                 d_ff,
                 self_attn_type,
                 dropout,
                 attention_dropout,
                 embeddings):
        super().__init__()
        self.transformer_layers = nn.ModuleList([DecoderLayer(d_model,
                                                            nhead,
                                                            d_ff,
                                                            dropout,
                                                            attention_dropout,
                                                            self_attn_type=self_attn_type)
                                                 for i in range(num_layers)])
        self.embeddings = embeddings
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    def forward(self, tgt, src_pad_mask, memory=None, step=None):
        # tgt : (B, tgt_len)
        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3 # (B, tgt_len, d_model)
        
        pad_idx = self.embeddings.word_padding_idx
        ## pad_idx와 같으면 True, 아니면 False
        tgt_pad_mask = tgt.data.eq(pad_idx).unsqueeze(1) # (B, 1, tgt_len)
        
        out = emb
        for layer in transformer_layers:
            out = layer(out,
                        memory,
                        src_pad_mask, # (B, 1, src_len)
                        tgt_pad_mask,
                        step=step)
        logits = self.layer_norm(out)
        return logits
        