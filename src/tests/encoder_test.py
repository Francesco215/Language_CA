import unittest
from src import Encoder,graph_initialization

import torch
import numpy as np

from src.tokenizer import Tokenizer
from src.positional_encoding import RotaryEncoding


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

    def test_sin_cos(self):
        positional_encoding=RotaryEncoding()
        n,h,d,e=50,4,100,2
        shape=(n,h,d,e)
        base=1e-5

        sin,cos=positional_encoding.make_sin_cos(shape)

        self.assertEqual(sin.shape[0],shape[0])
        self.assertEqual(sin.shape[1],shape[2])
        self.assertEqual(sin.shape,cos.shape)

        for _ in range(20):
            i0=np.random.randint(0,n)
            i1=np.random.randint(0,d)

            exponent=i1/(d-1)

            theta=torch.tensor(i0*base**exponent)

            expected_sin=torch.sin(theta)
            expected_cos=torch.cos(theta)

            self.assertTrue(torch.isclose(sin[i0,i1],expected_sin,1e-3,1e-3))
            self.assertTrue(torch.isclose(cos[i0,i1],expected_cos,1e-3,1e-3))