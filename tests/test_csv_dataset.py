import unittest
import os
from src.CSVDataset import *
from src.TokenizerPadder import *

class CSVDatasetTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.createTestFile()
        cls.filePath = "test_file.csv"

        cls.dataset = CSVDataset(cls.filePath)

    @classmethod
    def createTestFile(self):

        self.columns = "en,de\n"
        self.inputs = [
            "something in english",
            "something else in english",
            "and something else in english"
            ]

        self.targets = [
            "something in deutch",
            "something else in deutch",
            "and something else in deutch"
            ]

        with open("test_file.csv", "w") as f:

            f.write(self.columns)
            f.write(self.inputs[0] + "," + self.targets[0] + "\n")
            f.write(self.inputs[1] + "," + self.targets[1] + "\n")
            f.write(self.inputs[2] + "," + self.targets[2] + "\n")

    @classmethod
    def tearDownClass(cls):
        
        os.remove("test_file.csv")

    def test_can_get_len(self):

        self.assertEqual(len(self.dataset), len(self.inputs))

    def test_can_get_item(self):

        for idx in range(len(self.inputs)):

            self.assertEqual(self.dataset[idx], (self.inputs[idx], self.targets[idx]))

    def test_can_get_sample_for_tokenizer(self):

        sampleSize = 2
        tokenizer = TokenizerPadder('<|endofword|>', '<|endoftext|>')

        sample = self.dataset.getSampleForTokenizer(tokenizer, sampleSize)

        self.assertEqual(len(sample), 4)

        self.assertEqual(sample[0], tokenizer.addSpecialTokens(self.inputs[0]))
        self.assertEqual(sample[1], tokenizer.addSpecialTokens(self.targets[0]))
        self.assertEqual(sample[2], tokenizer.addSpecialTokens(self.inputs[1]))
        self.assertEqual(sample[3], tokenizer.addSpecialTokens(self.targets[1]))