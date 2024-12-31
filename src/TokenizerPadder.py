from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class CannotEncodeWithSmallerLenghtThanRealEncodingLenght(Exception):

    def __init__(self, lenghtGiven: int, lenghtReal: int):

        super().__init__(f"Can not encode with smaller lenght than real encoding lenght: given lenght is {lenghtGiven} but real lenght is {lenghtReal}")

class TokenizerPadder:

    def __init__(self, endOfWordToken: str, endOfTextToken: str):

        self.endOfWord = endOfWordToken
        self.endOfText = endOfTextToken

        self.tokenizer = Tokenizer(BPE())
        self.trainer = BpeTrainer(special_tokens=[endOfWordToken, endOfTextToken])

    def train(self, dataset: list[str]):
        
        self.tokenizer.train_from_iterator(dataset, self.trainer)

    def encode(self, input: str, lenght: int)-> list[int]:

        input = self.addSpecialTokens(input)
        encoded = self.tokenizer.encode(input).ids

        if lenght < len(encoded): 
            
            raise CannotEncodeWithSmallerLenghtThanRealEncodingLenght(lenght, len(encoded))

        for _ in range(len(encoded), lenght):

            encoded.append(self.tokenizer.encode(self.endOfText).ids[0])

        return encoded
    
    def cleanEOTPadding(self, input: list[int])-> list[int]:

        firstEOTidx = -1

        for idx in range(len(input)):

            if input[idx] == self.tokenizer.encode(self.endOfText).ids[0]:

                firstEOTidx = idx+1
                break

        input = input[:firstEOTidx]

        return input

    def cleanExtraSpaces(self, input: str)-> str:

        result = ""
        idx = 0
        while idx < len(input):

            if not input[idx] == " ":
                result += input[idx]

            else: 
                
                idx += 1
                if idx < len(input):
                    result += input[idx]

            idx += 1

        return result

    def decode(self, input: list[int])-> str:

        input = self.cleanEOTPadding(input)
        decodedWithSpace = self.tokenizer.decode(input, skip_special_tokens=False)
        result = self.cleanExtraSpaces(decodedWithSpace)

        return result
    
    def decodeBatch(self, input: list[str])-> list[list[int]]:

        result = []

        for tokens in input:

            result.append(self.decode(tokens))

        return result

    def encodeBatch(self, input: list[str], lenght: int)-> list[list[int]]:

        encoded = []

        for sentence in input:

            encoded.append(self.encode(sentence, lenght))

        return encoded
    
    def addEndOfWord(self, input: str)-> str:

        words = input.split(" ")

        for idx in range(len(words)):

            words[idx] += self.endOfWord

        result = " ".join(words)

        return result
    
    def addEndOfText(self, input: str)-> str:

        result = input + self.endOfText

        return result
    
    def addSpecialTokens(self, input: str)-> str:

        withEndOfWord = self.addEndOfWord(input)
        withEndOfText = self.addEndOfText(withEndOfWord)

        return withEndOfText