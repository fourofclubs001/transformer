from torch.utils.data import Dataset
import csv

class CSVDataset(Dataset):

    def __init__(self, filePath: str):

        self.filePath = filePath

        with open(filePath, "r") as file:

            reader = csv.DictReader(file)
            self.len = 0
            for row in reader: self.len += 1

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