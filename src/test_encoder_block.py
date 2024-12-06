from test_base import *
import torch
from EncoderBlock import *

class EncoderModuleTest(BaseTest):

    def setUp(self):

        super().setUp()

    def test_can_apply_attention(self):

        input = torch.ones((1, self.sequenceLenght, self.modelDimension))

        encoder = Encoder(self.modelDimension)
        output = encoder.applyAttention(input)

        expected = torch.ones((1, self.sequenceLenght, self.modelDimension))

        self.assertEqual(output.shape[0], expected.shape[0])
        self.assertEqual(output.shape[1], expected.shape[1])
        self.assertEqual(output.shape[2], expected.shape[2])

    def test_can_do_pass_forward(self):

        input = torch.ones((1, self.sequenceLenght, self.modelDimension))

        encoder = Encoder(self.modelDimension)
        output = encoder(input)

        expected = torch.ones((1, self.sequenceLenght, self.modelDimension))

        self.assertEqual(output.shape[0], expected.shape[0])
        self.assertEqual(output.shape[1], expected.shape[1])
        self.assertEqual(output.shape[2], expected.shape[2])