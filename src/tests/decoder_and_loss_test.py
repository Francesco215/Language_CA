import unittest
from src import Decoder, Loss

import torch

class DecoderTest(unittest.TestCase):
    def test_decoder(self):
        hidden_dim = 206
        embedding_dim = 103
        vocab_size = 523
        sequence_length = 10

        x = torch.rand(sequence_length, embedding_dim)
        decoder = Decoder(hidden_dim, embedding_dim, vocab_size)
        y = decoder(x)
        self.assertEqual(y.shape, (sequence_length, vocab_size))

class LossTest(unittest.TestCase):
    def test_loss(self):
        hidden_dim = 206
        embedding_dim = 103
        vocab_size = 523
        sequence_length = 10

        y=torch.randint(0,vocab_size,(sequence_length,))
        x = torch.rand(sequence_length, embedding_dim)
        decoder=Decoder(hidden_dim, embedding_dim, vocab_size)
        loss = Loss(decoder)
        y_hat = loss(x, y)
        self.assertEqual(y_hat.shape, torch.Size([]))


