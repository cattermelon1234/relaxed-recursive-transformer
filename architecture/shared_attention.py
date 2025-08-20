import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional

from transformer import MultiheadSelfAttention, MLP, TransformerLayer
from lora_layer import LoRALinear, LoRAAdapter, LoRAConv1D

class SharedAttention(nn.Module):
    def __init__(self, base_attn, num_repeats: int, lora_rank: int, lora_alpha: float):
        super().__init__()
        self.n_heads = base_attn.n_heads
        self.d_head = base_attn.d_head
        self.d_model = base_attn.d_model

        self.q_proj = LoRALinear(base_attn.q_proj, lora_rank, lora_alpha, num_repeats)
        self.k_proj = LoRALinear(base_attn.k_proj, lora_rank, lora_alpha, num_repeats)
        self.v_proj = LoRALinear(base_attn.v_proj, lora_rank, lora_alpha, num_repeats)
        self.out_proj = LoRALinear(base_attn.out_proj, lora_rank, lora_alpha, num_repeats)

    def forward(self, x, repeat_idx: int, attn_mask: Optional[torch.Tensor] = None):
        B, T, C = x.shape
        H, D = self.n_heads, self.d_head

        q = self.q_proj(x, repeat_idx).view(B, T, H, D).transpose(1,2)
        k = self.k_proj(x, repeat_idx).view(B, T, H, D).transpose(1,2)
        v = self.v_proj(x, repeat_idx).view(B, T, H, D).transpose(1,2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(D)
        if attn_mask is not None:
            att = att + attn_mask
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B, T, C)
        return self.out_proj(y, repeat_idx)

class SharedMLP(nn.Module):
    def __init__(self, base_mlp, num_repeats: int, lora_rank: int, lora_alpha: float):
        super().__init__()
        self.fc1 = LoRALinear(base_mlp.fc1, lora_rank, lora_alpha, num_repeats)
        self.fc2 = LoRALinear(base_mlp.fc2, lora_rank, lora_alpha, num_repeats)
        self.act = base_mlp.act

    def forward(self, x, repeat_idx: int):
        return self.fc2(self.act(self.fc1(x, repeat_idx)), repeat_idx)

class SharedTransformerLayer(nn.Module):
    def __init__(self, base_layer, num_repeats: int, lora_rank: int, lora_alpha: float):
        super().__init__()
        self.ln1 = base_layer.ln1
        self.ln2 = base_layer.ln2
        self.dropout1 = base_layer.dropout1
        self.dropout2 = base_layer.dropout2
        self.attn = SharedAttention(base_layer.attn, num_repeats, lora_rank, lora_alpha)
        self.mlp = SharedMLP(base_layer.mlp, num_repeats, lora_rank, lora_alpha)

    def forward(self, x, repeat_idx: int, attn_mask: Optional[torch.Tensor] = None):
        y = self.attn(self.ln1(x), repeat_idx, attn_mask)
        x = x + self.dropout1(y)
        y = self.mlp(self.ln2(x), repeat_idx)
        x = x + self.dropout2(y)
        return x

# ---- Conversion Utilities ----
def average_weights(layers, attr):
    weights = [getattr(layer, attr).weight.data for layer in layers]
    return torch.stack(weights, dim=0).mean(dim=0)


def initialize_lora_with_svd(lora_layer, original_weights, repeat_indices, rank):
    """
    original_weights: list of original weights for each repeat index
    repeat_indices: which repeat indices these weights correspond to
    """
    shared_weight = lora_layer.base_layer.weight.data.clone()
    
    for idx, orig_weight in zip(repeat_indices, original_weights):
        residual = orig_weight - shared_weight
        U, S, Vh = torch.linalg.svd(residual, full_matrices=False)
        
        # Truncate to rank
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]
        
        # Initialize LoRA weights
        lora_layer.lora_A[idx].weight.data = Vh  # A = Vᵣᵀ
        lora_layer.lora_B[idx].weight.data = U @ torch.diag(S)  # B = UᵣΣᵣ

def convert_to_recursive(model, K=2, rank=8, lora_alpha=1.0):
    n_layers = len(model.transformer.h)
    new_blocks = []
    
    for b in range(n_layers // K):
        block_layers = model.transformer.h[b*K:(b+1)*K]
        base_layer = copy.deepcopy(block_layers[0])
        
        # Average weights across the block for shared parameters
        with torch.no_grad():
            if hasattr(base_layer.attn, 'c_attn'):
                shared_weight = average_weights([l.attn for l in block_layers], 'c_attn')
                base_layer.attn.c_attn.weight.data = shared_weight
                
            if hasattr(base_layer.attn, 'c_proj'):
                shared_weight = average_weights([l.attn for l in block_layers], 'c_proj')
                base_layer.attn.c_proj.weight.data = shared_weight
                
            if hasattr(base_layer.mlp, 'c_fc'):
                shared_weight = average_weights([l.mlp for l in block_layers], 'c_fc')
                base_layer.mlp.c_fc.weight.data = shared_weight
                
            if hasattr(base_layer.mlp, 'c_proj'):
                shared_weight = average_weights([l.mlp for l in block_layers], 'c_proj')
                base_layer.mlp.c_proj.weight.data = shared_weight
        
        # Convert to LoRA
        if hasattr(base_layer.attn, 'c_attn'):
            base_layer.attn.c_attn = LoRAConv1D(
                base_layer.attn.c_attn, rank, lora_alpha, K
            )
        
        if hasattr(base_layer.attn, 'c_proj'):
            base_layer.attn.c_proj = LoRAConv1D(
                base_layer.attn.c_proj, rank, lora_alpha, K
            )
            
        if hasattr(base_layer.mlp, 'c_fc'):
            base_layer.mlp.c_fc = LoRAConv1D(
                base_layer.mlp.c_fc, rank, lora_alpha, K
            )
            
        if hasattr(base_layer.mlp, 'c_proj'):
            base_layer.mlp.c_proj = LoRAConv1D(
                base_layer.mlp.c_proj, rank, lora_alpha, K
            )
        
        new_blocks.append(base_layer)
    
    model.transformer.h = nn.ModuleList(new_blocks)
    return model