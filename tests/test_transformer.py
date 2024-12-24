from test_base import *
from src.Transformer import *

class TransformerTest(BaseTest):

    @classmethod
    def setUpClass(self):

        super().setUpClass()

        self.nTokens = 10
        self.transformer = Transformer(self.querySequenceLenght, self.nTokens)

    def test_can_apply_positional_encoding(self):

        output = self.transformer.applyPositionalEncoding(self.query)
        self.assert_equal_dimensions(output, self.query)

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

        sum = torch.sum(output, dim=1)

        self.assertTrue(0.9 < sum[0] and sum[0] < 1.1)
        self.assertTrue(0.9 < sum[1] and sum[1] < 1.1)

    def test_can_do_pass_forward(self):

        output = self.transformer(self.query, self.key)

        expected = torch.ones((self.query.shape[0], self.nTokens))
        self.assert_equal_dimensions(output, expected)

    def test_can_select_device_for_forward(self):

        device = torch.device('cuda')
        self.query = self.query.to(device)
        self.transformer.to(device)

        output = self.transformer(self.query, self.query)

        self.assertEqual(output.device.type, device.type)