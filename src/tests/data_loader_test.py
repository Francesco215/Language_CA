import unittest
from src.data_loader import Tokenizer

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