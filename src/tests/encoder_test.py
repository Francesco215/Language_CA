import unittest
from src import Encoder

import torch


hidden_dim=100
embedding_dim=100
base_freq=1e-2
dictionary_size=1000
sequence_length=130

encoder=Encoder(hidden_dim, embedding_dim, base_freq, dictionary_size)


class EncoderTest(unittest.TestCase):
    def test_encoder_shape(self):
        sequence_length=130
        x=torch.randint(0,dictionary_size-1,(sequence_length,))
        encoded=encoder(x)
        #self.assertEqual(2,2)
        self.assertEqual(encoded.shape, (sequence_length,embedding_dim))

    def test_values(self):
        x=torch.randint(0,dictionary_size*2,(sequence_length,))
        self.assertRaises(IndexError, encoder, x)

    def test_encoder_shape_batch(self):
        batch_size=10
        x=torch.randint(0,dictionary_size-1,(batch_size,sequence_length))
        encoded=encoder(x)
        #self.assertEqual(2,2)
        self.assertEqual(encoded.shape, (batch_size,sequence_length,embedding_dim))
    
    def test_values_batch(self):
        batch_size=10
        x=torch.randint(0,dictionary_size*2,(batch_size,sequence_length))
        self.assertRaises(IndexError, encoder, x)
    
    