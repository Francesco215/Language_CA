import unittest
from src import Decoder, Loss, Encoder

import torch

from src.data_loader import Tokenizer

class DecoderTest(unittest.TestCase):
    def test_decoder(self):
        tokenizer = Tokenizer("bert-base-cased", max_length=50)

        hidden_dim = 206
        embedding_dim = 103
        vocab_size = tokenizer.vocab_size
        sequence_length = 9

        x = torch.randn(sequence_length, embedding_dim)
        encoder = Encoder(embedding_dim, tokenizer)
        decoder = Decoder(encoder)
        y = decoder(x)
        self.assertEqual(y.shape, (sequence_length, vocab_size))

class LossTest(unittest.TestCase):
    def test_loss(self):
        tokenizer = Tokenizer("bert-base-cased", max_length=50)

        hidden_dim = 206
        embedding_dim = 103
        vocab_size = tokenizer.vocab_size
        sequence_length = 10

        y=torch.randint(0,vocab_size,(sequence_length,))
        x = torch.randn(sequence_length, embedding_dim)
        encoder = Encoder(embedding_dim, tokenizer)
        decoder = Decoder(encoder)
        loss = Loss(decoder)
        y_hat = loss(x, y)
        self.assertEqual(y_hat.shape, torch.Size([]))


