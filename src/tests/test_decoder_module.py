from test_base import *
from src.MultiHeadAttentionModule import *
from src.DecoderModule import *

class DecoderModuleTest(BaseTest):

    def setUp(self):

        super().setUp()

        self.decoderModule = DecoderModule(self.nHeads, self.modelDimension)

    def test_can_apply_mask_multi_head_attention(self):

        input = torch.ones((2, self.sequenceLenght, self.modelDimension))

        output = self.decoderModule.applyMaskMultiHeadAttention(input)
        self.assert_equal_dimensions(output, input)

    def test_can_apply_cross_multi_head_attention(self):

        output = self.decoderModule.applyCrossMultiHeadAttention(self.query, self.key)
        self.assert_equal_dimensions(output, self.query)

    def test_can_add_and_norm(self):

        layerNorm = nn.LayerNorm(self.modelDimension)
        output = self.decoderModule.addAndNorm(self.query,  self.query, layerNorm)
        self.assert_equal_dimensions(output, self.query)

    def test_can_do_pass_forward(self):

        output = self.decoderModule(self.query, self.key)
        self.assert_equal_dimensions(output, self.query)