import torch
import torch.nn as nn

CANNOT_USE_DIFFERENT_QUERY_AND_KEY_TOKEN_LENGHT_ERROR_MSG = "Cannot use different query and key token lenght"
CANNOT_USE_DIFFERENT_QUERY_AND_KEY_BATCH_LENGHT_ERROR_MSG = "Cannot use different query and key batch lenght"
CANNOT_FORWARD_WITH_DIFFERENT_KEY_AND_VALUE_SEQUENCE_LENGHT = "Cannot forward with different key and value sequence lenght"
QUERY_KEY__VALUE_TOKEN_LENGHT_MUST_MODEL_DIMENSION = "Query, Key and Value token lenght must match token dimension"

class CannotUseDifferentQueryAndKeyTokenLenght(Exception):

    def __init__(self):

        super().__init__(CANNOT_USE_DIFFERENT_QUERY_AND_KEY_TOKEN_LENGHT_ERROR_MSG)

class CannotUseDifferentQueryAndKeyBatchLenght(Exception):

    def __init__(self):

        super().__init__(CANNOT_USE_DIFFERENT_QUERY_AND_KEY_BATCH_LENGHT_ERROR_MSG)

class CannotForwardWithDifferentKeyValueSequenceLenght(Exception):

    def __init__(self):

        super().__init__(CANNOT_FORWARD_WITH_DIFFERENT_KEY_AND_VALUE_SEQUENCE_LENGHT)

class QueryKeyValueTokenLenghtMustMatchModelDimension(Exception):

    def __init__(self):

        super().__init__(QUERY_KEY__VALUE_TOKEN_LENGHT_MUST_MODEL_DIMENSION)

class AttentionModule(nn.Module):

    def __init__(self, modelDimension: int):

        super().__init__()

        self.modelDimension = modelDimension

        self.qW = nn.Linear(modelDimension, modelDimension)
        self.kW = nn.Linear(modelDimension, modelDimension)
        self.vW = nn.Linear(modelDimension, modelDimension)

    def checkSameKeyAndValueSequenceLenght(self, key: torch.Tensor, value: torch.Tensor)-> None:

        if key.shape[1] != value.shape[1]: raise CannotForwardWithDifferentKeyValueSequenceLenght

    def checkTokensLenght(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor)-> None:

        if query.shape[2]!= self.modelDimension: raise QueryKeyValueTokenLenghtMustMatchModelDimension
        if key.shape[2]!= self.modelDimension: raise QueryKeyValueTokenLenghtMustMatchModelDimension
        if value.shape[2]!= self.modelDimension: raise QueryKeyValueTokenLenghtMustMatchModelDimension

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor)-> torch.Tensor:

        self.checkSameKeyAndValueSequenceLenght(key, value)
        self.checkTokensLenght(query, key, value)

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