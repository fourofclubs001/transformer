from test_base import *
import torch
from EncoderBlock import *

class EncoderModuleTest(BaseTest):

    def setUp(self):

        super().setUp()

        self.encoder = Encoder(self.modelDimension)
        self.input = torch.ones((1, self.sequenceLenght, self.modelDimension))
        self.expected = torch.ones((1, self.sequenceLenght, self.modelDimension))

    def assert_equal_dimensions(self, result: torch.Tensor, expected: torch.Tensor)-> None:

        self.assertEqual(result.shape[0], expected.shape[0])
        self.assertEqual(result.shape[1], expected.shape[1])
        self.assertEqual(result.shape[2], expected.shape[2])

    def test_can_apply_attention(self):

        output = self.encoder.applyAttention(self.input)
        self.assert_equal_dimensions(output, self.expected)

    def test_can_do_pass_forward(self):

        output = self.encoder(self.input)
        self.assert_equal_dimensions(output, self.expected)