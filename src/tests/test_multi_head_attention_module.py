import torch
from test_base import *
from src.MultiHeadAttentionModule import *

class MultiHeadAttentionModuleTest(BaseTest):

    def setUp(self):

        super().setUp()

        self.multiHeadAttentionModule = MultiHeadAttentionModule(self.nHeads, self.modelDimension)

    def test_can_concatenate_heads_foward_pass(self):

        output = self.multiHeadAttentionModule.concatenateHeadsForwardPass(self.query, self.key, self.values)

        expected = torch.ones((2, self.querySequenceLenght, self.modelDimension*self.nHeads))

        self.assert_equal_dimensions(output, expected)

    def test_can_do_forward_pass(self):

        output = self.multiHeadAttentionModule(self.query, self.key, self.values)

        expected = torch.ones((2, self.querySequenceLenght, self.modelDimension))

        self.assert_equal_dimensions(output, expected)

    def test_can_do_mask_forward_pass(self):

        multiHeadAttentionModule = MultiHeadAttentionModule(self.nHeads, 
                                                            self.modelDimension, 
                                                            applyMask = True)

        output = multiHeadAttentionModule(self.query, self.key, self.values)

        expected = torch.ones((2, self.querySequenceLenght, self.modelDimension))

        self.assert_equal_dimensions(output, expected)