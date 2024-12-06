import torch
import torch.nn as nn
from Attention import *

class Encoder(nn.Module):

    def __init__(self, modelDimension: int):

        super().__init__()

        self.attentionBlock = AttentionBlock(modelDimension)
        self.linear = nn.Linear(modelDimension, modelDimension)
        self.layerNorm1 = nn.LayerNorm(modelDimension)
        self.layerNorm2 = nn.LayerNorm(modelDimension)

    def applyAttention(self, input: torch.Tensor)-> torch.Tensor:

        return self.attentionBlock(input, input, input)
    
    def forward(self, input: torch.Tensor)-> torch.Tensor:

        residual = input.clone()

        x = self.applyAttention(input)
        x += residual
        x = self.layerNorm1(x)

        residual = x.clone()

        x = self.linear(x)
        x += residual
        x = self.layerNorm2(x)

        return x