import torch
import torch.nn as nn

class AttentionBlock(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, x: torch.Tensor)-> torch.Tensor:

        return x
    
    def calculateCompatibility(self, 
                               query: torch.Tensor, 
                               key: torch.Tensor)-> torch.Tensor:

        keyTranspose = torch.transpose(key, 1, 2)
        compatibilityMatrix = torch.matmul(query, keyTranspose)

        return compatibilityMatrix

