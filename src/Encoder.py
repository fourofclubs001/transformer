import torch
from Attention import *

class Encoder:

    def __init__(self, tokenLenght: int, modelDimension: int):

        self.attentionBlock = AttentionBlock(tokenLenght, tokenLenght, modelDimension)

    def applyAttention(self, input: torch.Tensor)-> torch.Tensor:

        return self.attentionBlock(input, input, input)



