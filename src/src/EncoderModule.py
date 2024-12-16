import torch
import torch.nn as nn
from src.MultiHeadAttentionModule import *
from src.TransformerModule import *

class EncoderModule(TransformerModule):

    def __init__(self, nHeads: int, modelDimension: int):

        super().__init__(modelDimension, 2)

        self.multiHeadAttentionModule = MultiHeadAttentionModule(nHeads, modelDimension)

    def applyAttention(self, input: torch.Tensor)-> torch.Tensor:

        return self.multiHeadAttentionModule(input, input, input)

    def forward(self, input: torch.Tensor)-> torch.Tensor:

        residual = input.clone()

        x = self.applyAttention(input)
        x = self.addAndNorm(x, residual, self.layerNorms[0])

        residual = x.clone()

        x = self.linear(x)
        x = self.addAndNorm(x, residual, self.layerNorms[1])

        return x