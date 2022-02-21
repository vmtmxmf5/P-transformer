import torch
import torch.nn

from Encoder import *
from Decoder import *
from MHA import MultiHeadedAttention
from FeedForward import PositionwiseFeedForward
from PositionalEncoding import *


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 enc_num_layers,
                 dec_num_layers,
                 d_model,
                 nhead,
                 d_ff,
                 dropout,
                 attention_dropout,
                 pad_idx,
                 self_attn_type='scaled-dot'):
        super().__init__()
        self.src_embeddings = Embeddings(d_model, src_vocab_size, pad_idx)
        self.tgt_embeddings = Embeddings(d_model, tgt_vocab_size, pad_idx)
        self.encoder = Encoder(enc_num_layers, d_model, nhead, d_ff, dropout, attention_dropout, self.src_embeddings)
        self.decoder = Decoder(dec_num_layers, d_model, nhead, d_ff, self_attn_type, dropout, attention_dropout, self.tgt_embeddings)
        self.generator = nn.Sequential(nn.LayerNorm(d_model),
                                    nn.Linear(d_model, tgt_vocab_size),
                                    nn.LogSoftmax(dim=-1))
            
        
    def forward(self, src, tgt, lengths, step=None):
        dec_in = tgt[:, :-1]  # (B, tgt_len - 1)
        
        src_emb, memory, lengths, src_pad_mask = self.encoder(src, lengths)
        logits = self.decoder(dec_in, src_pad_mask, memory, src_pad_mask)
        return self.generator(logits)

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)