import torch
import torch.nn as nn
from src.EncoderModule import *
from src.DecoderModule import *
from src.PositionalEncoderModule import *

class Transformer(nn.Module):

    def __init__(self, sequenceLenght: int, nTokens: int):

        super().__init__()

        nHeads = 8
        modelDimension = 512
        nEncoders = 6
        nDecoders = 6

        self.embedder = nn.Embedding(nTokens, modelDimension)

        self.positionalEncoder = PositionalEncoderModule(modelDimension)

        self.encoders = nn.ModuleList([EncoderModule(nHeads, modelDimension) for _ in range(nEncoders)])
        self.decoders = nn.ModuleList([DecoderModule(nHeads, modelDimension) for _ in range(nDecoders)])

        self.linear = nn.Linear(modelDimension*sequenceLenght, nTokens)
        self.softmax = nn.Softmax(dim=1)

    def applyEmbedding(self, input: torch.Tensor)-> torch.Tensor:

        input = input.long()
        return self.embedder(input)

    def applyPositionalEncoding(self, input: torch.Tensor)-> torch.Tensor:

        return self.positionalEncoder(input)

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

    def forward(self, decoderInput: torch.Tensor, encoderInput: torch.Tensor)-> torch.Tensor:

        encoderInput = self.applyEmbedding(encoderInput)
        encoderInput = self.applyPositionalEncoding(encoderInput)
        encoderOutput = self.applyEncoders(encoderInput)

        decoderInput = self.applyEmbedding(decoderInput)
        decoderInput = self.applyPositionalEncoding(decoderInput)
        decoderOutput = self.applyDecoders(decoderInput, encoderOutput)
        
        output = self.applyOutput(decoderOutput)

        return output