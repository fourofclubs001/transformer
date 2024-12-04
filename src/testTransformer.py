import unittest
from transformer import *
import torch

class AttentionModuleTest(unittest.TestCase):

    def setUp(self):

        self.sequenceLenght = 8
        self.tokenLenght = 64

        self.onesTensor = torch.ones((1,self.sequenceLenght, self.tokenLenght))

        self.attentionBlock = AttentionBlock()

    def test_input_dimension_equals_output_dimension(self):

        input = self.onesTensor
        output = self.attentionBlock(input)

        self.assertEqual(input.size(), output.size())

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

        querySequenceLenght = self.sequenceLenght + 1
        keySequenceLenght = self.sequenceLenght

        compatibility = torch.rand((2, querySequenceLenght, keySequenceLenght))
        expectedCompatibility = torch.div(compatibility, torch.sqrt(torch.tensor(querySequenceLenght)))

        compatibility = self.attentionBlock.scaleCompatibility(compatibility)

        self.assertTrue(torch.eq(compatibility, expectedCompatibility).all())

    def test_can_apply_softmax_over_compatibility(self):

        querySequenceLenght = self.sequenceLenght + 1
        keySequenceLenght = self.sequenceLenght

        compatibility = torch.rand((2, querySequenceLenght, keySequenceLenght))

        compatibility = self.attentionBlock.softmaxCompatibility(compatibility)

        sumOne = torch.sum(compatibility, dim=2)
        expected = torch.ones((2, querySequenceLenght))
        vectorDifference = sumOne-expected
        diference = torch.sum(vectorDifference)

        self.assertTrue(diference < 1e-3)

    def test_can_calculate_output_tokens_using_values(self):

        querySequenceLenght = self.sequenceLenght + 1
        keySequenceLenght = self.sequenceLenght

        compatibility = torch.rand((2, querySequenceLenght, keySequenceLenght))

        valueSequenceLenght = self.sequenceLenght

        value = torch.ones((2,valueSequenceLenght, self.tokenLenght))

        output = self.attentionBlock.calculateOutput(compatibility, value)

        expected = compatibility @ value

        self.assertTrue(torch.eq(output, expected).all())

    def test_can_do_pass_foward(self):

        querySequenceLenght = self.sequenceLenght + 1
        keySequenceLenght = self.sequenceLenght

        queries = torch.ones((2, querySequenceLenght, self.tokenLenght))
        keys = torch.ones((2, keySequenceLenght, self.tokenLenght))
        values = torch.ones((2, keySequenceLenght, self.tokenLenght))

        output = self.attentionBlock(queries, keys, values)

        expected = torch.ones((2, querySequenceLenght, self.tokenLenght))

        self.assertTrue(torch.eq(output, expected).all())

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