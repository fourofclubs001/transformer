from test_base import *
from src.MultiHeadAttentionModule import *
from src.DecoderModule import *

class DecoderModuleTest(BaseTest):

    def test_can_apply_mask_multi_head_attention(self):

        decoderModule = DecoderModule(self.nHeads, self.modelDimension)

        input = torch.ones((2, self.sequenceLenght, self.modelDimension))

        output = decoderModule.applyMaskMultiHeadAttention(input)

        self.assert_equal_dimensions(output, input)

    def test_can_apply_cross_multi_head_attention(self):

        decoderModule = DecoderModule(self.nHeads, self.modelDimension)

        output = decoderModule.applyCrossMultiHeadAttention(self.query, self.key)

        self.assert_equal_dimensions(output, self.query)

    def test_can_do_pass_forward(self): pass