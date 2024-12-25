import unittest
import tiktoken
from src.EncoderPadder import *

class EncoderPadderTest(unittest.TestCase):

    def setUp(self):

        self.batch = ["something","something else"]
        self.simpleEncoder = tiktoken.encoding_for_model("gpt-4o")
        self.encoder = EncoderPadder(self.simpleEncoder)

        self.simpleEncode = self.simpleEncoder.encode_batch(self.batch)
        self.encode = self.encoder.encode_batch(self.batch)

    def test_encode_like_given_encoder(self):

        for senteceIdx in range(len(self.simpleEncode)):

            for tokenIdx in range(len(self.simpleEncode[senteceIdx])):

                self.assertEqual(self.encode[senteceIdx][tokenIdx],
                                 self.simpleEncode[senteceIdx][tokenIdx])
                
    def test_fill_with_padding_until_max_sequence_lenght(self):

        maxSequenceLenght = max([len(sentence) for sentence in self.simpleEncode])

        for senteceIdx in range(len(self.simpleEncode)):

            for tokenIdx in range(len(self.simpleEncode[senteceIdx]), maxSequenceLenght):

                self.assertEqual(self.encode[senteceIdx][tokenIdx], self.simpleEncoder.eot_token)

    def test_decode_like_given_encoder(self):

        simpleDecode = self.simpleEncoder.decode_batch(self.encode)
        decode = self.encoder.decode_batch(self.encode)

        self.assertEqual(simpleDecode, decode)
        