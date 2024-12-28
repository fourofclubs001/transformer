from torch.utils.data import Dataset
import redis
import csv
class RedisDataset(Dataset):

    def __init__(self, host: str, port: int):

        self.redisClient = redis.StrictRedis(
            host=host,
            port=port,
            decode_responses=True
        )

    def load(self, filePath: str, columns: list[str]):

        with open(filePath, "r") as file:

            reader = csv.DictReader(file)

            for idx, row in enumerate(reader):

                for column in columns:

                    self.redisClient.set(f"{column}_{idx}", row[column])

    def __len__(self):

        return len(self.redisClient.keys())//2