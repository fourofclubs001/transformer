import unittest
from src.TokenizerPadder import *

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class TokenizerPadderTest(unittest.TestCase):

    def setUp(self):

        self.lenght = 42

        self.endOfWordToken = "<|EOW|>"
        self.endOfTextToken = "<|EOT|>"

        self.tokenizer = Tokenizer(BPE())
        self.trainer = BpeTrainer(special_tokens=[self.endOfWordToken, self.endOfTextToken])

        self.dataset = ["This simple sentence.", "Another sentence not so simple."]

        self.tokenizer.train_from_iterator(self.dataset, self.trainer)

        self.tokenizerPadder = TokenizerPadder(self.endOfWordToken, self.endOfTextToken)
        self.tokenizerPadder.train(self.dataset)

    def test_can_encode_as_tokenizer(self):

        encoded = self.tokenizerPadder.addSpecialTokens(self.dataset[0])
        encoded = self.tokenizer.encode(encoded).ids

        encodedPadder = self.tokenizerPadder.encode(self.dataset[0], self.lenght)

        self.assertEqual(len(encodedPadder), self.lenght)

        for characterIdx in range(len(encoded)):

            self.assertEqual(encodedPadder[characterIdx], encoded[characterIdx])

    def test_can_add_end_of_word_token(self):

        sentence = self.tokenizerPadder.addEndOfWord(self.dataset[0])

        for word in sentence.split(" "):

            self.assertTrue(word.endswith(self.endOfWordToken))

    def test_can_add_end_of_text(self):

        sentence = self.tokenizerPadder.addEndOfText(self.dataset[0])

        self.assertTrue(sentence.endswith(self.endOfTextToken))

    def test_can_add_special_tokens(self):

        sentence = self.tokenizerPadder.addSpecialTokens(self.dataset[0])

        words = sentence.split(" ")

        for idx in range(len(words)-1):

            self.assertTrue(words[idx].endswith(self.endOfWordToken))

        self.assertTrue(sentence.endswith(self.endOfWordToken+self.endOfTextToken))

    def test_can_add_end_of_text_padding(self):

        encodedBatch = self.tokenizerPadder.encodeBatch(self.dataset, self.lenght)

        encoded0 = self.tokenizerPadder.encode(self.dataset[0], self.lenght)
        encoded1 = self.tokenizerPadder.encode(self.dataset[1], self.lenght)

        self.assertEqual(encodedBatch[0], encoded0)
        self.assertEqual(encodedBatch[1], encoded1)

    def test_can_decode(self):

        encoded = self.tokenizerPadder.encode(self.dataset[0], self.lenght)
        decoded = self.tokenizerPadder.decode(encoded)

        withSpecialTokens = self.tokenizerPadder.addSpecialTokens(self.dataset[0])

        self.assertEqual(decoded, withSpecialTokens)

    def test_decode_a_single_EOT(self):

        encoded = self.tokenizerPadder.encodeBatch(self.dataset, self.lenght)[0]
        decoded = self.tokenizerPadder.decode(encoded)

        withSpecialTokens = self.tokenizerPadder.addEndOfWord(self.dataset[0])
        withSpecialTokens += self.endOfTextToken

        self.assertEqual(decoded, withSpecialTokens)

    def test_can_decode_batch(self):

        encoded = self.tokenizerPadder.encodeBatch(self.dataset, self.lenght)
        decoded = self.tokenizerPadder.decodeBatch(encoded)

        withSpecialTokens = []

        for sentence in self.dataset:

            withSpecialTokens.append(self.tokenizerPadder.addSpecialTokens(sentence))

        for idx in range(len(decoded)):

            self.assertEqual(decoded[idx], withSpecialTokens[idx])

    def test_raise_error_when_lenght_lower_than_encoded_lenght(self):

        sentence = "This is the a sentence"

        encoded = self.tokenizerPadder.addSpecialTokens(sentence)
        encoded = self.tokenizer.encode(encoded)

        lenght = len(encoded)

        with self.assertRaises(CannotEncodeWithSmallerLenghtThanRealEncodingLenght) as error:

            self.tokenizerPadder.encode(sentence, lenght-1)

        self.assertEqual(error.exception.args[0], 
                         f"Can not encode with smaller lenght than real encoding lenght: given lenght is {lenght-1} but real lenght is {lenght}")