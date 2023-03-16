import unittest
from src import Encoder,graph_initialization

import torch

from src.data_loader import Tokenizer


#TODO: add some kind of test for the positional encoding

hidden_dim=100
embedding_dim=100
sequence_length=130

class EncoderTest(unittest.TestCase):
    def test_encoder_shape(self):
        tokenizer = Tokenizer("bert-base-cased", max_length=50)

        sequence_length=130
        vocab_size=tokenizer.vocab_size
        
        encoder=Encoder(embedding_dim, tokenizer)

        x=torch.randint(0,vocab_size-1,(sequence_length,))
        encoded=encoder(x)
        self.assertEqual(encoded.shape, (sequence_length,embedding_dim))

    def test_values(self):
        tokenizer = Tokenizer("bert-base-cased", max_length=50)
        sequence_length=130
        vocab_size=tokenizer.vocab_size
        
        encoder=Encoder(embedding_dim, tokenizer)

        x=torch.randint(0,vocab_size*2,(sequence_length,))
        self.assertRaises(IndexError, encoder, x)

    
    def test_values_batch(self):
        tokenizer = Tokenizer("bert-base-cased", max_length=50)
        sequence_length=130
        vocab_size=tokenizer.vocab_size
        batch_size=10

        encoder=Encoder(embedding_dim, tokenizer)

        x=torch.randint(0,vocab_size*2,(batch_size,sequence_length))
        self.assertRaises(IndexError, encoder, x)
    
    #this include the use of the graph
    def test_encoder_shape_graph(self):
        tokenizer = Tokenizer("bert-base-cased", max_length=50)
        sequence_length=130
        vocab_size=tokenizer.vocab_size
        encoder=Encoder(embedding_dim, tokenizer)
        x=torch.randint(0,vocab_size,(sequence_length,))
        edges=graph_initialization.random_graph_maker(window_width=1,avg_n_edges=5)(sequence_length)
        encoded=encoder(x)
        self.assertEqual(encoded.shape, (x.shape[0],embedding_dim))

    def test_encoded_type(self):
        tokenizer = Tokenizer("bert-base-cased", max_length=50)
        sequence_length=130
        vocab_size=tokenizer.vocab_size
        encoder=Encoder(embedding_dim, tokenizer)
        x=torch.randint(0,vocab_size,(sequence_length,))
        edges=graph_initialization.random_graph_maker(window_width=1,avg_n_edges=5)(sequence_length)
        encoded=encoder(x)
        self.assertEqual(type(encoded),torch.Tensor)