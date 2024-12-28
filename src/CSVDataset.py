from torch.utils.data import Dataset
import csv
from src.TokenizerPadder import *

class CSVDataset(Dataset):

    def __init__(self, filePath: str):

        self.filePath = filePath

        with open(filePath, "r") as file:

            reader = csv.DictReader(file)
            self.len = 0
            for row in reader:
                
                self.len += 1

    def __len__(self)-> int:

        return self.len
    
    def __getitem__(self, index):
        
        with open(self.filePath, "r") as file:

            reader = csv.DictReader(file)

            rowNumber = 0
            while rowNumber < index: 
                next(reader)
                rowNumber += 1

            row = next(reader)
            input = row["en"]
            target = row["de"]

        return input, target
    
    def getSampleForTokenizer(self, tokenizer: TokenizerPadder, nSamples: int)-> list[str]:

        samples = []

        with open(self.filePath, "r") as file:

            reader = csv.DictReader(file)

            for _ in range(nSamples):

                row = next(reader)
                input = row['en']
                target = row['de']

                input = tokenizer.addSpecialTokens(input)
                target = tokenizer.addSpecialTokens(target)

                samples.append(input)
                samples.append(target)

        return samples