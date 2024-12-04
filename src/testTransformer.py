import unittest
from transformer import *
import torch

class AttentionModuleTest(unittest.TestCase):

    def setUp(self):

        self.sequenceLenght = 8
        self.tokenLenght = 64

        self.onesTensor = torch.ones((1,self.sequenceLenght, self.tokenLenght))

        self.querySequenceLenght = self.sequenceLenght + 1
        self.keySequenceLenght = self.sequenceLenght

        queryKeyTokenLenght = self.tokenLenght
        valueTokenLenght = self.tokenLenght
        modelDimension = self.tokenLenght
        
        self.attentionBlock = AttentionBlock(queryKeyTokenLenght, valueTokenLenght, modelDimension)

    def test_can_calculate_compatibility(self):

        query = self.onesTensor.clone()
        key = self.onesTensor.clone()

        compatibility = self.attentionBlock.calculateCompatibility(query, key)

        expectedCompatibility = torch.ones((1, self.sequenceLenght, self.sequenceLenght))*64

        self.assertTrue(torch.eq(compatibility, expectedCompatibility).all())

    def test_can_calculate_compatibility_with_different_sequence_leght(self):

        query = self.onesTensor.clone()
        key = torch.ones((1, self.sequenceLenght+1, self.tokenLenght))

        compatibility = self.attentionBlock.calculateCompatibility(query, key)

        expectedCompatibility = torch.ones((1, self.sequenceLenght, self.sequenceLenght+1))*64

        self.assertTrue(torch.eq(compatibility, expectedCompatibility).all())

    def test_can_calculate_compatibility_with_greater_batch_size(self):

        query = torch.ones((2, self.sequenceLenght, self.tokenLenght))
        key = torch.ones((2, self.sequenceLenght, self.tokenLenght))

        compatibility = self.attentionBlock.calculateCompatibility(query, key)

        expectedCompatibility = torch.ones((2, self.sequenceLenght, self.sequenceLenght))*64

        self.assertTrue(torch.eq(compatibility, expectedCompatibility).all())

    def test_can_scale_compatibility_matrix(self):

        compatibility = torch.ones((2, self.querySequenceLenght, self.keySequenceLenght))
        expectedCompatibility = torch.div(compatibility, torch.sqrt(torch.tensor(self.querySequenceLenght)))

        compatibility = self.attentionBlock.scaleCompatibility(compatibility)

        self.assertTrue(torch.eq(compatibility, expectedCompatibility).all())

    def test_can_apply_softmax_over_compatibility(self):

        compatibility = torch.ones((2, self.querySequenceLenght, self.keySequenceLenght))

        compatibility = self.attentionBlock.softmaxCompatibility(compatibility)

        sumOne = torch.sum(compatibility, dim=2)
        expected = torch.ones((2, self.querySequenceLenght))

        self.assertTrue(torch.eq(sumOne, expected).all())

    def test_can_calculate_output_tokens_using_values(self):

        compatibility = torch.ones((2, self.querySequenceLenght, self.keySequenceLenght))

        valueSequenceLenght = self.sequenceLenght

        value = torch.ones((2,valueSequenceLenght, self.tokenLenght))

        output = self.attentionBlock.calculateOutput(compatibility, value)

        expected = compatibility @ value

        self.assertTrue(torch.eq(output, expected).all())

    def test_can_do_pass_foward(self):

        query = torch.ones((2, self.querySequenceLenght, self.tokenLenght))
        key = torch.ones((2, self.keySequenceLenght, self.tokenLenght))
        values = torch.ones((2, self.keySequenceLenght, self.tokenLenght))

        output = self.attentionBlock(query, key, values)

        expected = torch.ones((2, self.querySequenceLenght, self.tokenLenght))

        self.assertEqual(output.shape[0], expected.shape[0])
        self.assertEqual(output.shape[1], expected.shape[1])
        self.assertEqual(output.shape[2], expected.shape[2])

    def test_can_select_input_token_lenght(self):

        queryKeyTokenLenght = 8
        modelDimension = 16

        attentionBlock = AttentionBlock(queryKeyTokenLenght, self.tokenLenght, modelDimension)

        query = torch.ones((1, self.sequenceLenght, queryKeyTokenLenght))
        key = torch.ones((1, self.sequenceLenght, queryKeyTokenLenght))
        value = torch.ones((1, self.sequenceLenght, self.tokenLenght))

        output = attentionBlock(query, key, value)

        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[1], self.sequenceLenght)
        self.assertEqual(output.shape[2], modelDimension)

    def test_can_select_value_token_lenght(self):

        queryKeyTokenLenght = 8
        valueTokenLenght = 32
        modelDimension = 16

        attentionBlock = AttentionBlock(queryKeyTokenLenght, valueTokenLenght, modelDimension)

        query = torch.ones((1, self.sequenceLenght, queryKeyTokenLenght))
        key = torch.ones((1, self.sequenceLenght, queryKeyTokenLenght))
        value = torch.ones((1, self.sequenceLenght, valueTokenLenght))

        output = attentionBlock(query, key, value)

        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[1], self.sequenceLenght)
        self.assertEqual(output.shape[2], modelDimension)

    def assert_raise_error_calculate_compatibility(self,
                                                   query: torch.Tensor, 
                                                   key: torch.Tensor, 
                                                   exception: Exception, 
                                                   errorMessage: str):

        with self.assertRaises(exception) as error:

            self.attentionBlock.calculateCompatibility(query, key)

        self.assertEqual(error.exception.args[0], errorMessage)

    def test_raise_exception_when_different_token_size_on_compatibility(self): 

        query = self.onesTensor
        key = torch.ones((1, self.sequenceLenght, self.tokenLenght+1))

        self.assert_raise_error_calculate_compatibility(query, key, 
                                                        CannotUseDifferentQueryAndKeyTokenLenght, 
                                                        CANNOT_USE_DIFFERENT_QUERY_AND_KEY_TOKEN_LENGHT_ERROR_MSG)

    def test_raise_exception_when_different_number_of_batch_on_compatibility(self):

        query = self.onesTensor
        key = torch.ones((2, self.sequenceLenght, self.tokenLenght))

        self.assert_raise_error_calculate_compatibility(query, key, 
                                                        CannotUseDifferentQueryAndKeyBatchLenght, 
                                                        CANNOT_USE_DIFFERENT_QUERY_AND_KEY_BATCH_LENGHT_ERROR_MSG)
        
    def test_raise_exception_when_forward_with_different_key_and_value_sequence_lenght(self): pass

    def test_raise_exception_when_key_or_value_do_not_math_input_token_lenght(self): pass