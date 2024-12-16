import torch
import torch.nn as nn
from src.MultiHeadAttentionModule import *
from src.TransformerModule import *

class DecoderModule(TransformerModule):

    def __init__(self, nHeads: int, modelDimension: int):

        super().__init__(modelDimension)

        self.maskMultiHeadAttention = MultiHeadAttentionModule(nHeads, modelDimension, 
                                                               applyMask = True)

        self.crossMultiHeadAttention = MultiHeadAttentionModule(nHeads, modelDimension)

        self.layerNorm1 = nn.LayerNorm(modelDimension)
        self.layerNorm2 = nn.LayerNorm(modelDimension)
        self.layerNorm3 = nn.LayerNorm(modelDimension)

        self.linear = nn.Linear(modelDimension, modelDimension)

    def applyMaskMultiHeadAttention(self, input: torch.Tensor)-> torch.Tensor:

        return self.maskMultiHeadAttention(input, input, input)
    
    def applyCrossMultiHeadAttention(self, 
                                     decoderInput: torch.Tensor, 
                                     encoderOutput: torch.Tensor)-> torch.Tensor:
        
        return self.crossMultiHeadAttention(decoderInput, encoderOutput, encoderOutput)
    
    def forward(self, decoderInput: torch.Tensor, encoderOutput: torch.Tensor)-> torch.Tensor:

        residual = decoderInput.clone()

        x = self.applyMaskMultiHeadAttention(decoderInput)
        x = self.addAndNorm(x, residual, self.layerNorm1)

        residual = x.clone()

        x = self.applyCrossMultiHeadAttention(x, encoderOutput)
        x = self.addAndNorm(x, residual, self.layerNorm2)

        residual = x.clone()

        x = self.linear(x)
        x = self.addAndNorm(x, residual, self.layerNorm3)

        return x