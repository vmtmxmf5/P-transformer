import torch
import torch.nn as nn
import math

class MultiHeadedAttention(nn.Module):
    def __init__(self,
                nhead,
                d_model,
                dropout=0.1):
        assert d_model % nhead == 0
        super().__init__()
        self.head_dim = d_model // nhead
        self.d_model = d_model

        self.nhead = nhead
        self.Q = nn.Linear(d_model, nhead * self.head_dim)
        self.K = nn.Linear(d_model, nhead * self.head_dim)
        self.V = nn.Linear(d_model, nhead * self.head_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None, attn_type=None):
        # query : (B, query_len, d_model) // FloatTensor
        # key   : (B, key_len, d_model) // FloatTensor
        # value : (B, seq_len, d_model) // FloatTensor
        # mask  : (B, 1, src_len) == (B, query_len, key_len)
        #       : (B, tgt_len, src_len) == (B, query_len, key_len)

        B = key.size(0)
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            '''Projection'''
            # output : (B, nhead, seq_len, head_dim)
            return x.view(B, -1, self.nhead, self.head_dim).transpose(1, 2)
        
        def unshape(x):
            '''Compute context'''
            # output : (B, seq_len, d_model)
            return x.transpose(1, 2).contiguous().view(B, -1, self.nhead * self.head_dim)
        
        ## 1) Project Q, K, V
        Q = self.Q(query)
        K = self.K(key)
        V = self.V(value)
        K, V = shape(K), shape(V)

        Q = shape(Q) # Q : (B, nhead, query_len, head_dim)

        ## 2) Calculate scores
        Q = Q / math.sqrt(self.head_dim)
        # QK : (B, nhead, query_len, key_len)
        QK = torch.matmul(Q, K.transpose(2, 3))
        scores = QK.float()

        if mask is not None:
            mask = mask.unsqueeze(1) # (B, 1, 1 or query_len, key_len)
            scores = scores.masked_fill(mask, -1e18)
        
        ## 3) Apply dropout and compute context vector
        attn = self.softmax(scores).to(query.dtype) # (B, nhead, query_len, key_len)
        drop_attn = self.dropout(attn)

        # (B, nhead, query_len, key_len) @ (B, nhead, value_len, head_dim)

        context_original = torch.matmul(drop_attn, V)
        # (B, nhead, query_len, head_dim)

        context = unshape(context_original) # (B, q_len, d_model)
        output = self.final_linear(context) # (B, q_len, d_model)
        attns = attn.view(B, self.nhead, query_len, key_len)
        return output, attns
    
    def update_dropout(self, dropout):
        self.dropout.p = dropout
    


        

        
