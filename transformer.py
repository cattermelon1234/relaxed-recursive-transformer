import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Standard projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        B, T, C = x.shape
        H = self.n_heads
        D = self.d_head

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)   # (B, H, T, D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(D)         # (B, H, T, T)
        if attn_mask is not None:
            att = att + attn_mask  # mask should be broadcastable; use -inf on masked positions
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, H, T, D)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        return y

class MLP(nn.Module):  # Fixed: Now inherits from nn.Module
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor):
        return self.fc2(self.activation(self.fc1(x)))

class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.self_attn = MultiheadSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        y = self.self_attn(self.ln1(x), attn_mask)
        x = x + self.dropout(y)
        y = self.mlp(self.ln2(x))
        return x + self.dropout(y)

class Transformer(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(2048, d_model)  # simple fixed max length
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)  # Added missing final LayerNorm
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying
        
    def forward(self, idx: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for layer in self.layers:
            x = layer(x, attn_mask)
        x = self.ln_f(x)
        return self.lm_head(x)