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

    def load(self, filePath: str):

        with open(filePath, "r") as file:

            reader = csv.DictReader(file)

            for idx, row in enumerate(reader):

                self.redisClient.set(f"en_{idx}", row["en"])
                self.redisClient.set(f"de_{idx}", row["de"])