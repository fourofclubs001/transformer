from test_base import BaseTest
from src.PositionWiseFeedForwardModule import *

class PositionWiseFeedForwardModuleTest(BaseTest):

    def test_can_do_forward_pass(self):

        positionWiseFeedForward = PositionWiseFeedForwardModule(self.modelDimension, self.modelDimension*4)
        output = positionWiseFeedForward(self.query)
        self.assert_equal_dimensions(output, self.query)