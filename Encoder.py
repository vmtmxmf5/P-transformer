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

class EncoderLayer(nn.Module):
    def __init__(self,
                d_model,
                nhead,
                d_ff,
                dropout,
                attention_dropout):
        super().__init__()
        self.self_attn = MultiHeadedAttention(nhead,
                                            d_model,
                                            dropout=attention_dropout)
        self.feed_forward = PositionwiseFeedForward(d_model,
                                                    d_ff,
                                                    dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6) ## open-nmt 튜닝 기준
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        # input : (B, src_len, d_model)
        # mask : (B, 1, src_len)
        
        ## 1) Pre-LN + self attn + skip con.
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm,
                                    input_norm,
                                    input_norm,
                                    mask=mask,
                                    attn_type='self') ## 명시하는게 mask 실험시 편리
        # out : (B, src_len, d_model)
        out = self.dropout(context) + inputs
        
        ## 2) Pre-LN + FF + skip con.
        output = self.feed_forward(out)
        return output
    
    ### why 이거 쓰이기는 함?
    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout


class Encoder(nn.Module):
    def __init__(self,
                num_layers,
                d_model,
                nhead,
                d_ff,
                dropout,
                attention_dropout,
                embeddings):
        super().__init__()
        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [EncoderLayer(d_model,
                        nhead,
                        d_ff,
                        dropout,
                        attention_dropout)
            for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self, src, lengths=None):
        emb = self.embeddings(src) # (B, src_len, d_model)
        # MHA에서 (B, 1, 1, T_src)로 바꿀 예정
        mask = ~sequence_mask(lengths).unsqueeze(1) # (B, 1, src_len)
        out = emb
        for layer in self.transformer:
            out = layer(out, mask)
        out = self.layer_norm(out) # (B, src_len, d_model)
        return emb, out, lengths, mask

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)