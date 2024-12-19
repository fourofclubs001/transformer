import torch
import torch.nn as nn

class PositionWiseFeedForwardModule(nn.Module):

    def __init__(self, modelDimension: int, innerDimension: int):

        super().__init__()

        self.linear1 = nn.Linear(modelDimension, innerDimension)
        self.linear2 = nn.Linear(innerDimension, modelDimension)

    def forward(self, input: torch.Tensor)-> torch.Tensor:

        x = self.linear1(input)
        x = torch.relu(x)
        x = self.linear2(x)

        return x
