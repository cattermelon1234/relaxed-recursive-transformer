import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List

# ---- LoRA ----
class LoRAAdapter(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float = 1.0, 
                 weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        if rank > 0:
            self.A = nn.Parameter(torch.zeros((rank, in_features)))
            self.B = nn.Parameter(torch.zeros((out_features, rank)))
            
            # Initialize with SVD if base weight is provided
            if weight is not None:
                U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
                U = U[:, :rank]
                S = S[:rank]
                Vh = Vh[:rank, :]
                self.A.data = Vh  # (rank, in_features)
                self.B.data = U @ torch.diag(S)  # (out_features, rank)
            else:
                nn.init.normal_(self.A, std=1/rank)
                nn.init.zeros_(self.B)
        else:
            self.register_parameter('A', None)
            self.register_parameter('B', None)

    def delta(self) -> Optional[torch.Tensor]:
        if self.rank == 0 or self.A is None or self.B is None:
            return None
        return (self.B @ self.A) * (self.alpha / self.rank)  # (out, in)

    def lora_parameters(self):
        if self.A is not None:
            yield self.A
        if self.B is not None:
            yield self.B

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float = 1.0, num_repeats: int = 1):
        super().__init__()
        self.linear = linear  # base frozen linear
        self.rank = rank
        self.num_repeats = num_repeats

        if rank > 0:
            self.loras = nn.ModuleList([
                LoRAAdapter(linear.in_features, linear.out_features, rank, alpha)
                for _ in range(num_repeats)
            ])
        else:
            self.loras = nn.ModuleList([])

    def forward(self, x, repeat_idx: int = 0):
        out = self.linear(x)  # [batch, ..., out_features]
        if self.rank == 0:
            return out
        delta = self.loras[repeat_idx].delta()  # (out, in)
        if delta is not None:
            delta_t = delta  # nn.Linear expects (out, in)
            return out + F.linear(x, delta_t)
        return out

    def lora_parameters(self):
        for lora in self.loras:
            yield from lora.lora_parameters()


class LoRAConv1D(nn.Module):
    """GPT-2 style Conv1D with LoRA support."""
    def __init__(self, conv1d, rank: int, alpha: float = 1.0, num_repeats: int = 1):
        super().__init__()
        self.conv1d = conv1d  # base GPT-2 Conv1D
        self.rank = rank
        self.num_repeats = num_repeats
        in_features, out_features = conv1d.weight.shape  # GPT-2 Conv1D: [in, out]
        
        # Special handling for c_attn layer which has 3x output features
        self.is_c_attn = (out_features % 3 == 0) and ("c_attn" in str(conv1d))
        self.split_size = out_features // 3 if self.is_c_attn else out_features

        if rank > 0:
            if self.is_c_attn:
                # Create separate LoRA adapters for Q, K, V projections
                self.loras = nn.ModuleList([
                    nn.ModuleList([
                        LoRAAdapter(in_features, self.split_size, rank, alpha)
                        for _ in range(3)  # Q, K, V
                    ]) for _ in range(num_repeats)
                ])
            else:
                self.loras = nn.ModuleList([
                    LoRAAdapter(in_features, out_features, rank, alpha)
                    for _ in range(num_repeats)
                ])
        else:
            self.loras = nn.ModuleList([])

    def forward(self, x, repeat_idx: int = 0):
        """
        x: [batch, seq_len, in_features]
        returns: [batch, seq_len, out_features]
        """
        out = self.conv1d(x)
        if self.rank == 0 or len(self.loras) == 0:
            return out

        if self.is_c_attn:
            # Handle Q, K, V projections separately
            deltas = []
            for i in range(3):
                delta = self.loras[repeat_idx][i].delta()  # (split_size, in)
                if delta is not None:
                    delta_t = delta.T  # (in, split_size)
                    deltas.append(torch.matmul(x, delta_t))
            if deltas:
                return out + torch.cat(deltas, dim=-1)
            return out
        else:
            delta = self.loras[repeat_idx].delta()  # (out, in)
            if delta is not None:
                delta_t = delta.T  # (in, out)
                return out + torch.matmul(x, delta_t)
        return out
    
    def lora_parameters(self):
        if self.is_c_attn:
            for lora_group in self.loras:
                for lora in lora_group:
                    yield from lora.lora_parameters()
        else:
            for lora in self.loras:
                yield from lora.lora_parameters()