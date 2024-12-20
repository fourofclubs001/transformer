from test_base import *
from src.PositionalEncoderModule import *
import math

class PositionalEncodingModuleTest(BaseTest):

    def test_can_calculate_encoding_for_single_token(self):

        modelDimension = self.modelDimension

        def calculatePositionalEncoding(
                position: int, 
                dimension: int)-> torch.Tensor:

            value = position/(10000**(2*dimension/modelDimension))

            if dimension % 2 == 0: result = math.sin(value)
            else: result = math.cos(value)

            return result
        
        expected = [calculatePositionalEncoding(0,dimension) 
                   for dimension in range(self.modelDimension)]
        
        expected = torch.Tensor(expected)
        
        positionalEncoder = PositionalEncoderModule(self.modelDimension)
        output = positionalEncoder.calculatePositionalEncodingForSingleTokenAt(0)

        self.assertTrue(torch.equal(output, expected))
        

        
