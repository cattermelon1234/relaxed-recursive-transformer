import torch.nn.functional as F
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        # initialize A with variance 1/sqrt(rank) and B with zeros
        std_dev=1/torch.sqrt(torch.tensor(rank).float())
        self.A=nn.Parameter(torch.randn(in_dim, rank)*std_dev)
        self.B=nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha=alpha
        
    def forward(self, x):
        x=self.alpha*(x@self.A@self.B)
        return x

class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear=linear
        self.lora=LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )
    
    def forward(self, x):
        return self.linear(x)+self.lora(x)
    
def print_linear_layer_names(model):
    print("\n[Linear Layers in Model]")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(name)

def inject_lora(model, rank, alpha):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            lora_module = LinearWithLoRA(module, rank, alpha)
            parent_module = model
            for part in name.split('.')[:-1]:
                parent_module = getattr(parent_module, part)
            setattr(parent_module, name.split('.')[-1], lora_module)
    return model

def inject_lora_attention(model, rank, alpha):
    """
    Inject LoRA into only the attention query/value projection layers.
    Works for BERT-like transformer blocks.
    """
    # Print before injection
    print("Before injection:")
    print_linear_layer_names(model)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and (
            name.endswith("attention.self.query") or 
            name.endswith("attention.self.value")
        ):
            lora_module = LinearWithLoRA(module, rank, alpha)
            parent_module = model
            for part in name.split('.')[:-1]:
                parent_module = getattr(parent_module, part)
            setattr(parent_module, name.split('.')[-1], lora_module)

    # Print after injection
    print("\nAfter injection:")
    print_linear_layer_names(model)

    return model
