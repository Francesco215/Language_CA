import torch, unittest, transformers

gpt_tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
pretrained = transformers.GPT2LMHeadModel.from_pretrained('gpt2')



from src.GPT2 import GPT2, GPT2_Encoder, GPT2_LM_Head

class GPT2_loading_parameters(unittest.TestCase):
    def test_loading(self):
        Encoder=GPT2_Encoder(gpt_tokenizer)
        LM_Head=GPT2_LM_Head()
        gpt=GPT2(Encoder, LM_Head, gpt_tokenizer)

        gpt.load_from_original(pretrained)

        self.assertTrue(True)