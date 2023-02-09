import unittest
from src import Encoder,graph_initialization

import torch


#TODO: add some kind of test for the positional encoding

hidden_dim=100
embedding_dim=100
base_freq=1e-2
sequence_length=130

class EncoderTest(unittest.TestCase):
    def test_encoder_shape(self):
        sequence_length=130
        vocab_size=100
        
        encoder=Encoder(hidden_dim, embedding_dim, base_freq, vocab_size)

        x=torch.randint(0,vocab_size-1,(sequence_length,))
        encoded=encoder(x)
        self.assertEqual(encoded.shape, (sequence_length,embedding_dim))

    def test_values(self):
        sequence_length=130
        vocab_size=100
        
        encoder=Encoder(hidden_dim, embedding_dim, base_freq, vocab_size)

        x=torch.randint(0,vocab_size*2,(sequence_length,))
        self.assertRaises(IndexError, encoder, x)

    def test_encoder_shape_batch(self):
        #this test is not needed, but it is a good example of how to test a batch
        sequence_length=130
        vocab_size=100
        batch_size=10
        encoder=Encoder(hidden_dim, embedding_dim, base_freq, vocab_size)

        x=torch.randint(0,vocab_size-1,(batch_size,sequence_length))
        encoded=encoder(x)
        self.assertEqual(encoded.shape, (batch_size,sequence_length,embedding_dim))
    
    def test_values_batch(self):
        sequence_length=130
        vocab_size=100
        batch_size=10

        encoder=Encoder(hidden_dim, embedding_dim, base_freq, vocab_size)

        x=torch.randint(0,vocab_size*2,(batch_size,sequence_length))
        self.assertRaises(IndexError, encoder, x)
    
    #this include the use of the graph
    def test_encoder_shape_graph(self):
        sequence_length=130
        vocab_size=100
        encoder=Encoder(hidden_dim, embedding_dim, base_freq, vocab_size)
        x=torch.randint(0,vocab_size,(sequence_length,))
        edges=graph_initialization.random_graph_maker(window_width=1,avg_n_edges=5)(sequence_length)
        encoded=encoder(x)
        self.assertEqual(encoded.shape, (x.shape[0],embedding_dim))

    def test_encoded_type(self):
        sequence_length=130
        vocab_size=100
        encoder=Encoder(hidden_dim, embedding_dim, base_freq, vocab_size)
        x=torch.randint(0,vocab_size,(sequence_length,))
        edges=graph_initialization.random_graph_maker(window_width=1,avg_n_edges=5)(sequence_length)
        encoded=encoder(x)
        self.assertEqual(type(encoded),torch.Tensor)