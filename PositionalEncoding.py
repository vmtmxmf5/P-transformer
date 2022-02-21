import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, dropout, d_model, max_len=1000):
        assert d_model % 2 == 0, "d_model is odd"
        # Transformer PE : cos(pos / 10000^(2i/d_model)) or sin(pos / 10000^(2i/d_model))
        # EXP[-2i*log(10000)/d_model] == 10000^(-2i/d_model) == 1 / 10000^(2i/d_model)
        
        pe = torch.zeros(max_len, d_model)                    # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
                                                         
        div_term = torch.exp(                             # (d_model/2,)
            (torch.arange(0, d_model, 2, dtype=torch.float) * # (d_model/2,)
            -(math.log(10000.0) / d_model)))
        
        # (max_len, 1) * (d_model/2, ) = (max_len, d_model/2)
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)                              # (1, max_len, d_model/2)
        
        super(PositionalEncoding, self).__init__()
        # PositionalEncoding class를 pe layer로 등록
        # optimizer 업데이트 제외
        # state_dict()로 가중치 확인 가능
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, emb, step=None):
        # emb : (B, seq_len, d_model) // FloatTensor
        emb = emb * math.sqrt(self.d_model) # broadcasting
        step = step or 0 # step : search 시 다음 step에 pe 추가하기 위함 // int
        
        # max_len >= step + seq_len
        assert self.pe.size(1) >= step + emb.size(1), 'exceed max_len'

        # (B, seq_len, d_model) + (1, max_len => seq_len, dim/2)
        emb = emb + self.pe[:, step:emb.size(1)+step, :]
        emb = self.dropout(emb)
        return emb


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size, pad_idx, dropout=0, absolute_pe=True):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, pad_idx)
        self.absolute_pe = absolute_pe
        if self.absolute_pe:
            self.pe = PositionalEncoding(dropout, d_model)
            
    def forward(self, source, step=None):
        # source : (B, src_len) // LongTensor
        src_emb = self.embedding(source)
        if self.absolute_pe:
            src_emb = self.pe(src_emb, step=step)
        return src_emb