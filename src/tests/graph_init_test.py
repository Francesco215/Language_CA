import unittest

import torch, torch_geometric
from src import graph_initialization


converter=graph_initialization.text_to_graph()
batches=converter(["Hello world! good i saw you", "How are you?"])

class graph_init_test(unittest.TestCase):
    def test_graph_data_dtype(self):
        data=batches.get_example(0)
        x=data.x
        edge_index=data.edge_index
        self.assertEqual(x.dtype,torch.long)
        self.assertEqual(edge_index.dtype,torch.long)

    def test_graph_data_type(self):
        data=batches.get_example(0)
        x=data.x
        edge_index=data.edge_index
        self.assertEqual(type(x),torch.Tensor)
        self.assertEqual(type(edge_index),torch.Tensor)

    def test_graph_data_edge_shape(self):
        data=batches.get_example(0)
        edge_index=data.edge_index
        self.assertEqual(edge_index.shape[0],2)

    def test_data_range(self):
        data=batches.get_example(0)
        x=data.x
        edge_index=data.edge_index
        self.assertTrue(torch.all(x>=0))
        self.assertTrue(torch.all(edge_index>=0))
        self.assertTrue(torch.all(x<converter.vocab_size))
        self.assertTrue(torch.all(edge_index<converter.vocab_size))

    def test_batces_type(self):
        self.assertEqual(type(batches),torch_geometric.data.batch.DataBatch)
        self.assertEqual(type(batches[0]),torch_geometric.data.Data)
        data=batches.get_example(0)
        self.assertEqual(type(data),torch_geometric.data.Data)