from test_base import *
from src.PositionalEncoderModule import *
import math

class PositionalEncodingModuleTest(BaseTest):

    def calculate_positional_encoding(self, position: int, dimension: int)-> torch.Tensor:

            value = position/(10000**(2*dimension/self.modelDimension))

            if dimension % 2 == 0: result = math.sin(value)
            else: result = math.cos(value)

            return result

    def test_can_calculate_positional_encoding_for_single_token(self):
        
        expected = [self.calculate_positional_encoding(0,dimension) 
                   for dimension in range(self.modelDimension)]
        
        expected = torch.Tensor(expected)
        
        positionalEncoder = PositionalEncoderModule(self.modelDimension)
        output = positionalEncoder.calculatePositionalEncodingForSingleTokenAt(0)

        self.assertTrue(torch.equal(output, expected))

    def test_can_calculate_positional_encoding_for_sequence(self):
         
        expected = []

        for position in range(self.sequenceLenght):

            tokenExpected = [self.calculate_positional_encoding(position,dimension) 
                             for dimension in range(self.modelDimension)]  

            expected.append(tokenExpected)

        expected = torch.Tensor(expected)

        positionalEncoder = PositionalEncoderModule(self.modelDimension)
        output = positionalEncoder.calculateSequencePositionalEncoding(self.sequenceLenght)

        self.assertTrue(torch.equal(output, expected))

    def test_can_calculate_positional_encoding_for_batchs(self):

        batchLenght = 2
        batch = []

        for position in range(self.sequenceLenght):

            tokenExpected = [self.calculate_positional_encoding(position,dimension) 
                             for dimension in range(self.modelDimension)]  
            batch.append(tokenExpected)

        expected = [batch, batch]
        expected = torch.Tensor(expected)

        positionalEncoder = PositionalEncoderModule(self.modelDimension)
        output = positionalEncoder.calculateBatchesPositionalEncoding(batchLenght, self.sequenceLenght)

        self.assertTrue(torch.equal(output, expected))

    def test_can_do_pass_forward(self):
        
        batch = []

        for position in range(self.querySequenceLenght):

            tokenExpected = [self.calculate_positional_encoding(position,dimension) 
                             for dimension in range(self.modelDimension)]  
            batch.append(tokenExpected)

        expected = [batch, batch]
        expected = torch.Tensor(expected) + self.query

        positionalEncoder = PositionalEncoderModule(self.modelDimension)
        output = positionalEncoder(self.query)

        self.assertTrue(torch.equal(output, expected))

        

        
