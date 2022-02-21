import torch
import torch.nn

from MHA import MultiHeadedAttention
from FeedForward import PositionwiseFeedForward
from PositionalEncoding import *
from Encoder import *
from Decoder import *
from PEDec import *


class Ptransformer(nn.Module):
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
        self.p_tf = PEDec(enc_num_layers,
                        dec_num_layers,
                        d_model, nhead,
                        d_ff,
                        self_attn_type,
                        dropout,
                        attention_dropout,
                        self.src_embeddings,
                        self.tgt_embeddings)
        self.generator = nn.Sequential(nn.LayerNorm(d_model),
                                    nn.Linear(d_model, tgt_vocab_size),
                                    nn.LogSoftmax(dim=-1))
            
        
    def forward(self, src, tgt, lengths, step=None):
        dec_in = tgt[:, :-1]  # (B, tgt_len - 1)
        
        logits, _ = self.p_tf(src, tgt, lengths, step)
        return self.generator(logits)

    def update_dropout(self, dropout):
        self.p_tf.update_dropout(dropout)


if __name__=='__main__':
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    src_vocab_size = 10                 
    tgt_vocab_size = 10
    enc_num_layers = 6
    dec_num_layers = 6
    d_model = 512
    nhead = 8
    d_ff = 1024
    dropout = 0.1
    attention_dropout = 0.1
    pad_idx = 1
    model = Ptransformer(src_vocab_size,
                        tgt_vocab_size,
                        enc_num_layers,
                        dec_num_layers,
                        d_model,
                        nhead,
                        d_ff,
                        dropout,
                        attention_dropout,
                        pad_idx)

    # Transformer Info.
    # for layer in model.state_dict():
    #     print(layer, '\t', model.state_dict()[layer].size())
    
    # Prediction
    src_tmp = torch.LongTensor([[4, 6, 7, 8, 3, 1],
                                [7, 6, 5, 9, 4, 3]])
    tgt_tmp = torch.LongTensor([[5, 5, 7, 8, 3, 1],
                                [7, 4, 5, 4, 4, 3]])
    lengths = torch.LongTensor([5, 6])
    
    pred = model(src_tmp, tgt_tmp, lengths)
    print(pred)
    