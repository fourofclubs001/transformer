import torch
import torch.nn as nn
from src.AttentionModule import *

class MultiHeadAttentionModule(nn.Module):

    def __init__(self, nHeads: int, modelDimension: int):

        super().__init__()

        self.linear = nn.Linear(modelDimension*nHeads, modelDimension)

        self.attentionHeads = []

        for _ in range(nHeads):

            self.attentionHeads.append(AttentionModule(modelDimension))

    def concatenateHeadsForwardPass(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor)-> torch.Tensor:

        headsResults = []

        for attentionHead in self.attentionHeads:

            headsResults.append(attentionHead(query, key, value))

        result = torch.concat(headsResults, dim=2)

        return result

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor)-> torch.Tensor:

        concatenation = self.concatenateHeadsForwardPass(query, key, value)
        result = self.linear(concatenation)

        return result

