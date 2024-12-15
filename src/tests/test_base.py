import unittest
import torch

class BaseTest(unittest.TestCase):

    def setUp(self):

        self.sequenceLenght = 8
        self.modelDimension = 512

        self.querySequenceLenght = self.sequenceLenght + 1
        self.keySequenceLenght = self.sequenceLenght

        self.query = torch.ones((2, self.querySequenceLenght, self.modelDimension))
        self.key = torch.ones((2, self.keySequenceLenght, self.modelDimension))
        self.values = torch.ones((2, self.keySequenceLenght, self.modelDimension))

        self.nHeads = 8

    def assert_equal_dimensions(self, result: torch.Tensor, expected: torch.Tensor)-> None:

        self.assertEqual(result.shape[0], expected.shape[0])
        self.assertEqual(result.shape[1], expected.shape[1])
        self.assertEqual(result.shape[2], expected.shape[2])