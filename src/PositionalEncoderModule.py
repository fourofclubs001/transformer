import torch
import torch.nn as nn
import math

class PositionalEncoderModule(nn.Module):

    def __init__(self, modelDimension: int):

        super().__init__()

        self.modelDimension = modelDimension

    def positionEncoding(self, position: int, dimension: int)-> float:

            value = position/(10000**(2*dimension/self.modelDimension))

            if dimension % 2 == 0: result = math.sin(value)
            else: result = math.cos(value)

            return result

    def calculatePositionalEncodingForSingleTokenAt(self, position: int)-> torch.Tensor:

        result = [self.positionEncoding(position, dimension) for dimension in range(self.modelDimension)]

        result = torch.Tensor(result)

        return result
    
    def calculateSequencePositionalEncoding(self, sequenceLenght: int)-> torch.Tensor:
         
        result = torch.zeros((sequenceLenght, self.modelDimension))

        for position in range(sequenceLenght):
             
            result[position] = self.calculatePositionalEncodingForSingleTokenAt(position)

        return result
    
    def calculateBatchesPositionalEncoding(self, batchLenght: int, sequenceLenght: int)-> torch.Tensor:
         
        result = torch.zeros((batchLenght, sequenceLenght, self.modelDimension))

        for batch in range(batchLenght):
              
            result[batch] = self.calculateSequencePositionalEncoding(sequenceLenght)

        return result
    
    def forward(self, input: torch.Tensor)-> torch.Tensor:
         
        output = input + self.calculateBatchesPositionalEncoding(input.shape[0], input.shape[1])

        return output
