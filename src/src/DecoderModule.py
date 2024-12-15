import torch
import torch.nn as nn
from src.MultiHeadAttentionModule import *

class DecoderModule(nn.Module):

    def __init__(self, nHeads: int, modelDimensions: int):

        super().__init__()

        self.maskMultiHeadAttention = MultiHeadAttentionModule(nHeads, modelDimensions, 
                                                               applyMask = True)

        self.crossMultiHeadAttention = MultiHeadAttentionModule(nHeads, modelDimensions)

    def applyMaskMultiHeadAttention(self, input: torch.Tensor)-> torch.Tensor:

        return self.maskMultiHeadAttention(input, input, input)
    
    def applyCrossMultiHeadAttention(self, 
                                     decoderInput: torch.Tensor, 
                                     encoderInput: torch.Tensor)-> torch.Tensor:
        
        return self.crossMultiHeadAttention(decoderInput, encoderInput, encoderInput)