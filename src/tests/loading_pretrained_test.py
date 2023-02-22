import torch, unittest, transformers
from src import Tokenizer

pretrained = transformers.GPT2LMHeadModel.from_pretrained('gpt2')



from src.GPT2 import GPT2, GPT2_Encoder, GPT2_LM_Head
from src import linear_unidirectional_graph_maker

class GPT2_loading_parameters(unittest.TestCase):
    def test_loading(self):
        tokenizer = Tokenizer('gpt2')

        Encoder=GPT2_Encoder()
        LM_Head=GPT2_LM_Head()
        model=GPT2(Encoder, LM_Head, tokenizer)

        model.load_from_original(pretrained)

        
        self.assertTrue(True)


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

