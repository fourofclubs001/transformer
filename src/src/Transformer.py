import torch
import torch.nn as nn
from src.EncoderModule import *
from src.DecoderModule import *

class Transformer(nn.Module):

    def __init__(self, sequenceLenght: int, nTokens: int):

        super().__init__()

        nHeads = 8
        modelDimension = 512
        nEncoders = 6
        nDecoders = 6

        self.encoders = [EncoderModule(nHeads, modelDimension) for _ in range(nEncoders)]
        self.decoders = [DecoderModule(nHeads, modelDimension) for _ in range(nDecoders)]

        self.linear = nn.Linear(modelDimension*sequenceLenght, nTokens)
        self.softmax = nn.Softmax()

    def applyEncoders(self, input: torch.Tensor)-> torch.Tensor:

        x = input
        for encoder in self.encoders: x = encoder(x)

        return x
    
    def applyDecoders(self, input: torch.Tensor, encoderOutput: torch.Tensor)-> torch.Tensor:

        x = input
        for decoder in self.decoders: x = decoder(x, encoderOutput)

        return x
    
    def applyOutput(self, input: torch.Tensor)-> torch.Tensor:

        x = torch.reshape(input, (input.shape[0], input.shape[1]*input.shape[2]))

        x = self.linear(x)
        x = self.softmax(x)

        return x