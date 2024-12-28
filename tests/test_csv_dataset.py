import unittest
import os
from src.CSVDataset import *

class CSVDatasetTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.createTestFile()
        filePath = "test_file.csv"

        cls.dataset = CSVDataset(filePath)

    @classmethod
    def createTestFile(self):

        self.columns = "en,de\n"
        self.inputs = [
            "something in english",
            "something else in english"
            ]

        self.targets = [
            "something in deutch",
            "something else in deutch"
            ]

        with open("test_file.csv", "w") as f:

            f.write(self.columns)
            f.write(self.inputs[0] + "," + self.targets[0] + "\n")
            f.write(self.inputs[1] + "," + self.targets[1] + "\n")

    @classmethod
    def tearDownClass(cls):
        
        os.remove("test_file.csv")

    def test_can_get_len(self):

        self.assertEqual(len(self.dataset), 2)

    def test_can_get_item(self):

        self.assertEqual(self.dataset[0], (self.inputs[0], self.targets[0]))
        self.assertEqual(self.dataset[1], (self.inputs[1], self.targets[1]))