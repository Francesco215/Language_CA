import unittest
from src.data_loader import Tokenizer,Wiki
from src.graph_initialization import batch_graphs,random_graph_maker
import torch


class Tokenizer_test(unittest.TestCase):
    def test_tokenizer(self):
        text="This is a test"

        tokenizer=Tokenizer(max_length=10)
        out=tokenizer(text)
        
        self.assertEqual(out.dim(), 1)
        self.assertEqual(type(out), torch.Tensor)
        self.assertEqual(out.dtype, torch.int64)


    def test_tokenizer_length(self):
        text="This is a test"

        tokenizer1=Tokenizer(max_length=10)
        out1=tokenizer1(text)
        
        tokenizer2=Tokenizer(max_length=50)
        out2=tokenizer2(text)

        assertion=(out1==out2).all()
        self.assertTrue(assertion)

    def test_tokenizer_on_list(self):
        text=["This is a test","This is another test but with different length"]

        tokenizer=Tokenizer(max_length=50)
        out=tokenizer(text)
        
        self.assertEqual(type(out), list)
        self.assertEqual(type(out[0]), torch.Tensor)
        self.assertEqual(out[0].dtype, torch.int64)


class Wiki_test(unittest.TestCase):
    def test_dataloader(self):
        dummy_dataset=[
            {'text':"This is a test"},
            {'text':"This is another test but with different length"},
            {'text':"This is the last test string, i promise no more strings"}
        ]

        tokenizer=Tokenizer(max_length=50)
        graph_maker=random_graph_maker(1,5)
        data=Wiki(dummy_dataset,tokenizer,graph_maker)

        nodes,edges=data[0]
        self.assertEqual(type(nodes), torch.Tensor)
        self.assertEqual(type(edges), torch.Tensor)

    def test_data_loader_slice(self):
        dummy_dataset=[
            {'text':"This is a test"},
            {'text':"This is another test but with different length"},
            {'text':"This is the last test string, i promise no more strings"}
        ]

        tokenizer=Tokenizer(max_length=50)
        graph_maker=random_graph_maker(1,5)
        data=Wiki(dummy_dataset,tokenizer,graph_maker)

        nodes,edges=data[0:2]

        self.assertEqual(type(nodes), list)
        self.assertEqual(type(edges), list)
        self.assertEqual(type(nodes[0]), torch.Tensor)
        self.assertEqual(type(edges[0]), torch.Tensor)

    def test_data_loader_slice_batched(self):
        dummy_dataset=[
            {'text':"This is a test"},
            {'text':"This is another test but with different length"},
            {'text':"This is the last test string, i promise no more strings"}
        ]

        tokenizer=Tokenizer(max_length=50)
        graph_maker=random_graph_maker(1,5)
        data=Wiki(dummy_dataset,tokenizer,graph_maker)

        nodes,edges=data[0:2]

        nodes,edges=batch_graphs(nodes,edges)

        self.assertEqual(type(nodes), torch.Tensor)
        self.assertEqual(type(edges), torch.Tensor)