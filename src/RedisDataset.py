from torch.utils.data import Dataset
import redis
import csv
class RedisDataset(Dataset):

    def __init__(self, host: str, port: int, firstColumn: str, secondColumn: str):

        self.redisClient = redis.StrictRedis(
            host=host,
            port=port,
            decode_responses=True
        )

        self.firstColumn = firstColumn
        self.secondColumn = secondColumn

    def load(self, filePath: str):

        with open(filePath, "r") as file:

            reader = csv.DictReader(file)

            for idx, row in enumerate(reader):

                self.redisClient.set(f"{self.firstColumn}_{idx}", row[self.firstColumn])
                self.redisClient.set(f"{self.secondColumn}_{idx}", row[self.secondColumn])

    def __len__(self):

        return len(self.redisClient.keys())//2
    
    def __getitem__(self, index)-> tuple[str]:

        firstResult = self.redisClient.get(f"{self.firstColumn}_{index}")
        secondResult = self.redisClient.get(f"{self.secondColumn}_{index}")

        return firstResult, secondResult