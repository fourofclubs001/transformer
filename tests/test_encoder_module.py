from test_base import *
import torch
from src.EncoderModule import *

class EncoderModuleTest(BaseTest):

    def setUp(self):

        super().setUp()

        self.encoderModule = EncoderModule(self.nHeads, self.modelDimension)

        self.input = torch.ones((1, self.sequenceLenght, self.modelDimension))
        self.expected = torch.ones((1, self.sequenceLenght, self.modelDimension))

    def test_can_apply_attention(self):

        output = self.encoderModule.applyAttention(self.input)
        self.assert_equal_dimensions(output, self.expected)

    def test_can_do_pass_forward(self):

        output = self.encoderModule(self.input)
        self.assert_equal_dimensions(output, self.expected)