import torch
import torch.nn as nn
from src.PositionWiseFeedForwardModule import *

class TransformerModule(nn.Module):

    def __init__(self, modelDimension: int, nLayerNorm: int, innerPositionWiseFeedForwardDimension: int = None):

        super().__init__()

        if innerPositionWiseFeedForwardDimension == None:
        
            innerPositionWiseFeedForwardDimension = 4*modelDimension

        self.linear = PositionWiseFeedForwardModule(modelDimension, innerPositionWiseFeedForwardDimension)

        self.layerNorms = []

        for _ in range(nLayerNorm):

            self.layerNorms.append(nn.LayerNorm(modelDimension))

        self.layerNorms = nn.ModuleList(self.layerNorms)

    def addAndNorm(self, 
                   current: torch.Tensor, 
                   residual: torch.Tensor, 
                   layerNorm: nn.LayerNorm)-> torch.Tensor:
        
        current += residual
        current = layerNorm(current)

        return current