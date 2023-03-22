import unittest
from src import Decoder, Loss, Encoder

import torch

from src.tokenizer import Tokenizer

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


    def test_encoder_decoder(self):
        tokenizer = Tokenizer("bert-base-cased", max_length=50)

        hidden_dim = 206
        embedding_dim = 103
        vocab_size = tokenizer.vocab_size
        sequence_length = 9


        encoder = Encoder(embedding_dim, tokenizer)
        decoder = Decoder(encoder)
        
        
        x = torch.randint(0,vocab_size, (sequence_length,))
        y = decoder(encoder(x))

        self.assertTrue((y.argmax(dim=-1)==x).all())

class LossTest(unittest.TestCase):
    def test_loss(self):
        tokenizer = Tokenizer("bert-base-cased", max_length=50)

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

    def test_cross_entropy_function(self):
        tokenizer = Tokenizer("bert-base-cased", max_length=50)

        embedding_dim = 103
        vocab_size = tokenizer.vocab_size
        sequence_length = 10

        
        encoder = Encoder(embedding_dim, tokenizer)
        decoder = Decoder(encoder)

        y = torch.randint(0,vocab_size,(sequence_length,))
        x = torch.randn(sequence_length, embedding_dim)

        loss=Loss(decoder)
        l1 = loss(x, y)


        def categorical_cross_entropy(x, target):
            assert target.dtype == torch.int64


            logsoftmax=torch.nn.LogSoftmax(dim=-1)

            log_pred=logsoftmax(decoder(x))
            target=torch.nn.functional.one_hot(target, num_classes=vocab_size)
            loss = -torch.mean((target * log_pred).sum(dim=-1))
            
            return loss
        

        l2 = categorical_cross_entropy(x, y)
        self.assertTrue(torch.isclose(l2, l1, 1e-3, 1e-3))

    
