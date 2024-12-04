import torch
import torch.nn as nn

CANNOT_USE_DIFFERENT_QUERY_AND_KEY_TOKEN_LENGHT_ERROR_MSG = "Cannot use different query and key token lenght"
CANNOT_USE_DIFFERENT_QUERY_AND_KEY_BATCH_LENGHT_ERROR_MSG = "Cannot use different query and key batch lenght"

class CannotUseDifferentQueryAndKeyTokenLenght(Exception):

    def __init__(self):

        super().__init__(CANNOT_USE_DIFFERENT_QUERY_AND_KEY_TOKEN_LENGHT_ERROR_MSG)

class CannotUseDifferentQueryAndKeyBatchLenght(Exception):

    def __init__(self):

        super().__init__(CANNOT_USE_DIFFERENT_QUERY_AND_KEY_BATCH_LENGHT_ERROR_MSG)

class AttentionBlock(nn.Module):

    def __init__(self, queryKeyTokenLenght: int, valueTokenLenght: int, modelDimension: int):

        super().__init__()

        self.qW = nn.Linear(queryKeyTokenLenght, modelDimension)
        self.kW = nn.Linear(queryKeyTokenLenght, modelDimension)
        self.vW = nn.Linear(valueTokenLenght, modelDimension)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor)-> torch.Tensor:

        query = self.qW(query)
        key = self.kW(key)
        value = self.vW(value)
        
        compatibility = self.calculateCompatibility(query, key)
        compatibility = self.scaleCompatibility(compatibility)
        compatibility = self.softmaxCompatibility(compatibility)
        output = self.calculateOutput(compatibility, value)

        return output
    
    def checkQueryKeyDimensionCompatibility(self, query: torch.Tensor, key: torch.Tensor)-> None:

        if query.shape[0] != key.shape[0]: raise CannotUseDifferentQueryAndKeyBatchLenght
        if query.shape[2] != key.shape[2]: raise CannotUseDifferentQueryAndKeyTokenLenght

    def calculateCompatibility(self, 
                               query: torch.Tensor, 
                               key: torch.Tensor)-> torch.Tensor:

        self.checkQueryKeyDimensionCompatibility(query, key)

        keyTranspose = torch.transpose(key, 1, 2)
        compatibilityMatrix = torch.matmul(query, keyTranspose)

        return compatibilityMatrix
    
    def scaleCompatibility(self, compatibility: torch.Tensor)-> torch.Tensor:

        return torch.div(compatibility, torch.sqrt(torch.tensor(compatibility.shape[1])))
    
    def softmaxCompatibility(self, compatibility: torch.Tensor)-> torch.Tensor:

        return torch.softmax(compatibility, dim=2)
    
    def calculateOutput(self, compatibility: torch.Tensor, value: torch.Tensor)-> torch.Tensor:

        return torch.matmul(compatibility, value)