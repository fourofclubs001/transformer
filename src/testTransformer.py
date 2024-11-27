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

    def test_can_calculate_output_tokens_using_values(self): pass

    def test_raise_exception_when_different_token_size_on_compatibility(self): pass

    def test_raise_exception_when_different_number_of_batch_on_compatibility(self): pass