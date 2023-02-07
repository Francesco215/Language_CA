import unittest
from src import Encoder,graph_initialization

import torch,torch_geometric


hidden_dim=100
embedding_dim=100
base_freq=1e-2
sequence_length=130
converter=graph_initialization.text_to_graph()
dictionary_size=converter.vocab_size

encoder=Encoder(hidden_dim, embedding_dim, base_freq, converter.vocab_size)

batches=converter(["Hello world! good i saw you", "How are you?"])
data=batches.get_example(0)
class EncoderTest(unittest.TestCase):
    def test_encoder_shape(self):
        sequence_length=130
        x=torch.randint(0,dictionary_size-1,(sequence_length,))
        encoded=encoder(x)
        self.assertEqual(encoded.shape, (sequence_length,embedding_dim))

    def test_values(self):
        x=torch.randint(0,dictionary_size*2,(sequence_length,))
        self.assertRaises(IndexError, encoder, x)

    def test_encoder_shape_batch(self):
        batch_size=10
        x=torch.randint(0,dictionary_size-1,(batch_size,sequence_length))
        encoded=encoder(x)
        self.assertEqual(encoded.shape, (batch_size,sequence_length,embedding_dim))
    
    def test_values_batch(self):
        batch_size=10
        x=torch.randint(0,dictionary_size*2,(batch_size,sequence_length))
        self.assertRaises(IndexError, encoder, x)
    
    #this include the use of the graph
    def test_encoder_shape_graph(self):
        x=data.x
        edge_index=data.edge_index
        encoded=encoder(x)
        self.assertEqual(encoded.shape, (x.shape[0],embedding_dim))

    def test_encoded_type(self):
        x=data.x
        edge_index=data.edge_index
        encoded=encoder(x)
        self.assertEqual(type(encoded),torch.Tensor)