import unittest
import redis
import os
from src.RedisDataset import *
from src.TokenizerPadder import *

class RedisDatasetTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.create_data()
        cls.create_test_csv_file()
        cls.create_redis_client()
        cls.clean_database()
        cls.create_dataset()
        
    @classmethod
    def create_data(cls):

        cls.englishColumn = "en"
        cls.deutchColumn = "de"

        cls.deutchSenteces = [
            "something in deutch",
            "something else in deutch",
            "another thing in deutch"
            ]
        
        cls.englishSenteces = [
            "something in english",
            "something else in english",
            "another thing in english"
            ]

    @classmethod
    def create_test_csv_file(cls):

        cls.testDatasetFilePath = "test_dataset.csv"

        with open(cls.testDatasetFilePath, "w") as file:

            file.write(cls.deutchColumn + "," + cls.englishColumn + "\n")

            for idx in range(len(cls.englishSenteces)):
                file.write(cls.deutchSenteces[idx] + "," + cls.englishSenteces[idx] + "\n")

    @classmethod
    def create_redis_client(cls):

        cls.host = "redis-testing"
        cls.port = 6379

        cls.redisClient = redis.StrictRedis(host=cls.host,port=cls.port,
                                            decode_responses=True)

    @classmethod
    def clean_database(cls):

        for key in cls.redisClient.keys():

            cls.redisClient.delete(key)

    @classmethod
    def create_dataset(cls):

        cls.prefixTrain = "train"
        cls.prefixTest = "test"

        cls.trainDataset = RedisDataset(cls.host, cls.port, cls.prefixTrain, cls.englishColumn, cls.deutchColumn)
        cls.trainDataset.load(cls.testDatasetFilePath)

        cls.testDataset = RedisDataset(cls.host, cls.port, cls.prefixTest, cls.englishColumn, cls.deutchColumn)
        cls.testDataset.load(cls.testDatasetFilePath)

    @classmethod
    def tearDownClass(cls):
        
        os.remove(cls.testDatasetFilePath)

        for key in cls.redisClient.keys():

            cls.redisClient.delete(key)

    def test_can_load_dataset(self):
        
        for idx in range(len(self.englishSenteces)):

            self.assertEqual(self.redisClient.get(f"{self.prefixTrain}_{self.englishColumn}_{idx}"), 
                                                  self.englishSenteces[idx])
            self.assertEqual(self.redisClient.get(f"{self.prefixTrain}_{self.deutchColumn}_{idx}"), 
                                                  self.deutchSenteces[idx])
            
            self.assertEqual(self.redisClient.get(f"{self.prefixTest}_{self.englishColumn}_{idx}"), 
                                                  self.englishSenteces[idx])
            self.assertEqual(self.redisClient.get(f"{self.prefixTest}_{self.deutchColumn}_{idx}"), 
                                                  self.deutchSenteces[idx])

    def test_can_get_len(self):

        self.assertEqual(len(self.trainDataset), len(self.englishSenteces))
        self.assertEqual(len(self.testDataset), len(self.englishSenteces))

    def test_can_get_item(self):

        for idx in range(len(self.englishSenteces)):

            self.assertEqual(self.trainDataset[idx], (self.englishSenteces[idx], self.deutchSenteces[idx]))
            self.assertEqual(self.testDataset[idx], (self.englishSenteces[idx], self.deutchSenteces[idx]))