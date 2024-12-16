import torch
import torch.nn as nn

class TransformerModule(nn.Module):

    def __init__(self, modelDimension: int):

        super().__init__()

        self.linear = nn.Linear(modelDimension, modelDimension)

    def addAndNorm(self, 
                   current: torch.Tensor, 
                   residual: torch.Tensor, 
                   layerNorm: nn.LayerNorm)-> torch.Tensor:
        
        current += residual
        current = layerNorm(current)

        return current