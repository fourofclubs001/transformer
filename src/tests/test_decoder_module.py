from test_base import *
from src.MultiHeadAttentionModule import *
from src.DecoderModule import *

class DecoderModuleTest(BaseTest):

    def test_can_apply_mask_multi_head_attention(self):

        decoderModule = DecoderModule()

        input = torch.ones((2, self.sequenceLenght, self.modelDimension))

        output = decoderModule.applyMaskMultiHeadAttention(input)

        self.assert_equal_dimensions(output, input)

    def test_can_apply_cross_multi_head_attention(self): pass

    def test_can_do_pass_forward(self): pass