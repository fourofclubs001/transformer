import torch
import torch.nn as nn

class TransformerModule(nn.Module):

    def __init__(self):

        super().__init__()

    def addAndNorm(self, 
                   current: torch.Tensor, 
                   residual: torch.Tensor, 
                   layerNorm: nn.LayerNorm)-> torch.Tensor:
        
        current += residual
        current = layerNorm(current)

        return current