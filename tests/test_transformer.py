from test_base import *
from src.Transformer import *

class TransformerTest(BaseTest):

    def setUp(self):

        super().setUp()

        self.nTokens = 10
        self.transformer = Transformer(self.querySequenceLenght, self.nTokens)

    def test_can_apply_encoders(self):

        output = self.transformer.applyEncoders(self.query)
        self.assert_equal_dimensions(output, self.query)

    def test_can_apply_decoders(self):

        output = self.transformer.applyDecoders(self.query, self.query)
        self.assert_equal_dimensions(output, self.query)

    def test_can_apply_output(self):

        output = self.transformer.applyOutput(self.query)

        expected = torch.ones((self.query.shape[0], self.nTokens))
        self.assert_equal_dimensions(output, expected)

    def test_can_do_pass_forward(self):

        output = self.transformer(self.query, self.key)

        expected = torch.ones((self.query.shape[0], self.nTokens))
        self.assert_equal_dimensions(output, expected)