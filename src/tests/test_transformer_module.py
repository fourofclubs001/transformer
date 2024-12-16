import torch.nn as nn
from test_base import *
from src.TransformerModule import *

class TransformerModuleTest(BaseTest):

    def test_can_add_and_norm(self):

        transformerModule = TransformerModule()

        layerNorm = nn.LayerNorm(self.modelDimension)
        output = transformerModule.addAndNorm(self.query,  self.query, layerNorm)
        self.assert_equal_dimensions(output, self.query)