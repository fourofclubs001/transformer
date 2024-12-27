import unittest
from src.TokenizerPadder import *

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class TokenizerPadderTest(unittest.TestCase):

    def setUp(self):

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

        encodedPadder = self.tokenizerPadder.encode(self.dataset[0])

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

        encodedBatch = self.tokenizerPadder.encodeBatch(self.dataset)

        encodedShort = self.tokenizerPadder.encode(self.dataset[0])
        encodedShortWithPadding = encodedBatch[0]

        encodedLong = self.tokenizerPadder.encode(self.dataset[1])
        encodedLongWithPadding = encodedBatch[1]

        self.assertEqual(encodedLong, encodedLongWithPadding)

        for idx in range(len(encodedShortWithPadding)):

            if idx < len(encodedShort):

                self.assertEqual(encodedShort[idx], encodedShortWithPadding[idx])

            else:

                endOfTextTokenNumber = self.tokenizer.encode(self.endOfTextToken).ids[0]
                self.assertEqual(endOfTextTokenNumber, encodedShortWithPadding[idx])

    def test_can_decode(self):

        encoded = self.tokenizerPadder.encode(self.dataset[0])
        decoded = self.tokenizerPadder.decode(encoded)

        withSpecialTokens = self.tokenizerPadder.addSpecialTokens(self.dataset[0])

        self.assertEqual(decoded, withSpecialTokens)

    def test_decode_a_single_EOT(self):

        encoded = self.tokenizerPadder.encodeBatch(self.dataset)[0]
        decoded = self.tokenizerPadder.decode(encoded)

        withSpecialTokens = self.tokenizerPadder.addEndOfWord(self.dataset[0])
        withSpecialTokens += self.endOfTextToken

        self.assertEqual(decoded, withSpecialTokens)

    def test_can_decode_batch(self):

        encoded = self.tokenizerPadder.encodeBatch(self.dataset)
        decoded = self.tokenizerPadder.decodeBatch(encoded)

        withSpecialTokens = []

        for sentence in self.dataset:

            withSpecialTokens.append(self.tokenizerPadder.addSpecialTokens(sentence))

        for idx in range(len(decoded)):

            self.assertEqual(decoded[idx], withSpecialTokens[idx])