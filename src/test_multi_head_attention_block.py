import unittest
import torch
from test_base import *
from MultiHeadAttentionBlock import *

class MultiHeadAttentionModuleTest(BaseTest):

    def setUp(self):

        super().setUp()

        self.multiHeadAttentionBlock = MultiHeadAttentionBlock(self.nHeads, self.modelDimension)

    def test_can_concatenate_heads_foward_pass(self):

        query = torch.ones((2, self.querySequenceLenght, self.modelDimension))
        key = torch.ones((2, self.keySequenceLenght, self.modelDimension))
        values = torch.ones((2, self.keySequenceLenght, self.modelDimension))

        output = self.multiHeadAttentionBlock.concatenateHeadsForwardPass(query, key, values)

        expected = torch.ones((2, self.querySequenceLenght, self.modelDimension*self.nHeads))

        self.assert_equal_dimensions(output, expected)

    def test_can_do_forward_pass(self):

        query = torch.ones((2, self.querySequenceLenght, self.modelDimension))
        key = torch.ones((2, self.keySequenceLenght, self.modelDimension))
        values = torch.ones((2, self.keySequenceLenght, self.modelDimension))

        output = self.multiHeadAttentionBlock(query, key, values)

        expected = torch.ones((2, self.querySequenceLenght, self.modelDimension))

        self.assert_equal_dimensions(output, expected)