import unittest
import torch
from test_base import *
from MultiHeadAttentionBlock import *

class MultiHeadAttentionModuleTest(BaseTest):

    def setUp(self):

        super().setUp()

        self.querySequenceLenght = self.sequenceLenght + 1
        self.keySequenceLenght = self.sequenceLenght

        self.nHeads = 8
        self.multiHeadAttentionBlock = MultiHeadAttentionBlock(self.nHeads, self.modelDimension)

    def test_can_concatenate_heads_foward_pass(self):

        query = torch.ones((2, self.querySequenceLenght, self.modelDimension))
        key = torch.ones((2, self.keySequenceLenght, self.modelDimension))
        values = torch.ones((2, self.keySequenceLenght, self.modelDimension))

        output = self.multiHeadAttentionBlock.concatenateHeadsForwardPass(query, key, values)

        expected = torch.ones((2, self.querySequenceLenght, self.modelDimension*self.nHeads))

        self.assertEqual(output.shape[0], expected.shape[0])
        self.assertEqual(output.shape[1], expected.shape[1])
        self.assertEqual(output.shape[2], expected.shape[2])

    def test_can_do_forward_pass(self):

        query = torch.ones((2, self.querySequenceLenght, self.modelDimension))
        key = torch.ones((2, self.keySequenceLenght, self.modelDimension))
        values = torch.ones((2, self.keySequenceLenght, self.modelDimension))

        output = self.multiHeadAttentionBlock(query, key, values)

        expected = torch.ones((2, self.querySequenceLenght, self.modelDimension))

        self.assertEqual(output.shape[0], expected.shape[0])
        self.assertEqual(output.shape[1], expected.shape[1])
        self.assertEqual(output.shape[2], expected.shape[2])