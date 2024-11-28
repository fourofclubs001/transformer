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

    def __init__(self):

        super().__init__()

    def forward(self, x: torch.Tensor)-> torch.Tensor:

        return x
    
    def calculateCompatibility(self, 
                               query: torch.Tensor, 
                               key: torch.Tensor)-> torch.Tensor:

        if query.shape[0] != key.shape[0]:

            raise CannotUseDifferentQueryAndKeyBatchLenght

        if query.shape[2] != key.shape[2]:

            raise CannotUseDifferentQueryAndKeyTokenLenght

        keyTranspose = torch.transpose(key, 1, 2)
        compatibilityMatrix = torch.matmul(query, keyTranspose)

        return compatibilityMatrix

