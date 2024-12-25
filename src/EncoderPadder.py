import tiktoken

class EncoderPadder:

    def __init__(self, encoder: tiktoken.core.Encoding):

        self.encoder = encoder

    def encode_batch(self, batch: list[str])-> list[list[int]]:

        encode = self.encoder.encode_batch(batch)
        maxSequenceLenght = max([len(sequence) for sequence in encode])

        for sequence in encode:

            for _ in range(len(sequence), maxSequenceLenght):

                sequence.append(self.encoder.eot_token)

        return encode
    
    def decode_batch(self, encodedBatch: list[list[int]])-> list[str]:

        return self.encoder.decode_batch(encodedBatch)

