import torch
import torch.nn as nn

from MHA import MultiHeadedAttention
from FeedForward import PositionwiseFeedForward
from PositionalEncoding import *
from Encoder import EncoderLayer, sequence_mask
from Decoder import DecoderLayer


class subclass(nn.Module):
    def __init__(self,
                d_model,
                nhead,
                d_ff,
                self_attn_type,
                dropout,
                attention_dropout):
        super().__init__()
        self.enc = EncoderLayer(d_model,
                                nhead,
                                d_ff,
                                dropout,
                                attention_dropout)
        self.dec = DecoderLayer(d_model,
                                nhead,
                                d_ff,
                                dropout,
                                attention_dropout,
                                self_attn_type=self_attn_type)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
    def forward(self, src_out, src_mask, tgt_out, tgt_pad_mask, step=None):
        # TODO LayerNorm?? Memory 여러개가 각기 다르게 변한다면, 업데이트마다 기존 Transformer보다 변동성이 크다
        # 그러므로 반드시! LayerNorm을 해줘야 학습이 용이하다
        # Encoder
        src_out = self.enc(src_out, src_mask)
        src_out = self.layer_norm(src_out)
        # Decoder
        tgt_out, attn = self.dec(tgt_out,
                                src_out,
                                src_mask, # (B, 1, src_len)
                                tgt_pad_mask,
                                step=step)
        return tgt_out

class PEDec(nn.Module):
    def __init__(self,
                enc_num_layers,
                dec_num_layers,
                d_model,
                nhead,
                d_ff,
                self_attn_type,
                dropout,
                attention_dropout,
                enc_embeddings,
                dec_embeddings):
        super().__init__()
        self.enc_embeddings = enc_embeddings
        self.dec_embeddings = dec_embeddings
        self.pre_tf = enc_num_layers > dec_num_layers
        self.post_tf = enc_num_layers < dec_num_layers
        if self.pre_tf:
            self.pre_transformer = nn.ModuleList(
                [EncoderLayer(d_model,
                            nhead,
                            d_ff,
                            dropout,
                            attention_dropout)
                for i in range(enc_num_layers - dec_num_layers)]) 
            enc_num_layers = dec_num_layers
            
            self.transformer = nn.ModuleList(
                [subclass(d_model,
                        nhead,
                        d_ff,
                        self_attn_type,
                        dropout,
                        attention_dropout)
                for i in range(enc_num_layers)])
            
        elif self.post_tf:
            self.transformer = nn.ModuleList(
                [subclass(d_model,
                        nhead,
                        d_ff,
                        self_attn_type,
                        dropout,
                        attention_dropout)
                for i in range(enc_num_layers)])

            self.post_transformer = nn.ModuleList(
                [DecoderLayer(d_model,
                            nhead,
                            d_ff,
                            dropout,
                            attention_dropout,
                            self_attn_type=self_attn_type)
                for i in range(dec_num_layers - enc_num_layers)])
        else:
            self.transformer = nn.ModuleList(
                [subclass(d_model,
                        nhead,
                        d_ff,
                        self_attn_type,
                        dropout,
                        attention_dropout)
                for i in range(enc_num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self, src, tgt, lengths=None, step=None):
        src_emb = self.enc_embeddings(src) # (B, src_len, d_model)
        # MHA에서 (B, 1, 1, T_src)로 바꿀 예정
        src_mask = ~sequence_mask(lengths).unsqueeze(1) # (B, 1, src_len)
        tgt_emb = self.dec_embeddings(tgt, step=step)
        assert tgt_emb.dim() == 3 # (B, tgt_len, d_model)
        
        pad_idx = self.dec_embeddings.pad_idx
        ## pad_idx와 같으면 True, 아니면 False
        tgt_pad_mask = tgt.data.eq(pad_idx).unsqueeze(1) # (B, 1, tgt_len)
        
        src_out, tgt_out = src_emb, tgt_emb

        if self.pre_tf:
            for pre_layer in self.pre_transformer:
                src_out = pre_layer(src_out, src_mask)

        for layer in self.transformer:
            tgt_out = layer(src_out, src_mask, tgt_out, tgt_pad_mask, step)
            
        if self.post_tf:
            for post_layer in self.post_transformer:
                tgt_out = post_layer(tgt_out,
                                src_out,
                                src_mask, # (B, 1, src_len)
                                tgt_pad_mask,
                                step=step)
        out = self.layer_norm(tgt_out) # (B, src_len, d_model)        
        return out, lengths

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)

        
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
