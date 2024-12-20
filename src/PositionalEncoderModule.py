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
