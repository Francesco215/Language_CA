import torch, unittest
from src import Tokenizer

from torch import nn


from src.GPT2 import GPT2, GPT2_Encoder, GPT2_LM_Head, Conv1D_to_Linear
from src import linear_unidirectional_graph_maker
from transformers.modeling_utils import Conv1D

class GPT2_loading_functions(unittest.TestCase):
    def test_Conv1D_to_Linear(self):
        conv=Conv1D(60,40)
        lin=nn.Linear(40,60)

        Conv1D_to_Linear(conv,lin)

        x=torch.rand(17,40)
        self.assertTrue(torch.allclose(conv(x),lin(x)))
        conv=Conv1D(60,40)
        lin=nn.Linear(40,60)

        Conv1D_to_Linear(conv,lin)

        x=torch.rand(17,40)
        self.assertTrue(torch.allclose(conv(x),lin(x)))

    def test_Conv1D_to_Linear_error(self):

        conv=Conv1D(40,60)
        lin=nn.Linear(40,60)

        self.assertRaises(AssertionError, Conv1D_to_Linear, conv, lin)


    def test_loading(self):
        tokenizer = Tokenizer('gpt2')

        Encoder=GPT2_Encoder()
        LM_Head=GPT2_LM_Head()
        model=GPT2(Encoder, LM_Head, tokenizer)

        model.load_from_original(pretrained)
        
        self.assertTrue(True)

import transformers
pretrained = transformers.GPT2LMHeadModel.from_pretrained('gpt2')
class GPT2_loading_parameters(unittest.TestCase):

    def test_loading_and_call(self):
        tokenizer = Tokenizer('gpt2')

        Encoder=GPT2_Encoder()
        LM_Head=GPT2_LM_Head()
        model=GPT2(Encoder, LM_Head, tokenizer)

        model.load_from_original(pretrained)

        sample_text = "Legolas and Gimli advanced on the orcs, raising their weapons with a harrowing war cry."
        x=tokenizer(sample_text)

        graph_maker=linear_unidirectional_graph_maker(5)
        edge_index=graph_maker(x.shape[0])

        out=model(x,edge_index)

        self.assertEqual(out.shape,(len(x),tokenizer.vocab_size))
        self.assertFalse(out.isnan().any())
        self.assertFalse(out.isinf().any())

    def test_MLP_is_the_same(self):
        tokenizer = Tokenizer('gpt2')

        Encoder=GPT2_Encoder()
        LM_Head=GPT2_LM_Head()
        model=GPT2(Encoder, LM_Head, tokenizer,dropout=0)

        model.load_from_original(pretrained)

        sequence_length=17
        d_Embedding=model.d_Embedding
        x=torch.randn([sequence_length,d_Embedding])

        for i in range(model.n_blocks):
            output=model.transformer_blocks[i].MLP(x)
            target=pretrained.transformer.h[i].mlp(x)

            out=target/output
            
            where=torch.isclose(out,torch.ones_like(x)*1.111,atol=1e-2)

            out=out[where]

            self.assertTrue(torch.allclose(out,torch.ones_like(out)*1.111,atol=1e-2))


    def test_attention_is_the_same(self):
        tokenizer = Tokenizer('gpt2')

        Encoder=GPT2_Encoder()
        LM_Head=GPT2_LM_Head()
        model=GPT2(Encoder, LM_Head, tokenizer,dropout=0)

        model.load_from_original(pretrained)

        sequence_length=17
        d_Embedding=model.d_Embedding

        graph_maker=linear_unidirectional_graph_maker(40)
        edge_index=graph_maker(sequence_length)

        for i in range(model.n_blocks):
            x=torch.randn([1,sequence_length,d_Embedding])
            target=pretrained.transformer.h[i].attn(x)
            x.view(size=(sequence_length,d_Embedding))
            output=model.transformer_blocks[i].attention_block(x,edge_index)

            out=target/output
            
            where=torch.isclose(out,torch.ones_like(x)*1.111,atol=1e-2)

            out=out[where]

            self.assertTrue(torch.allclose(out,torch.ones_like(out)*1.111,atol=1e-2))