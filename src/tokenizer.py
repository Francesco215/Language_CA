
import torch
from transformers import AutoTokenizer


class Tokenizer:

    def __init__(self,
                 tokenizer="bert-base-cased",
                 max_length=512,
                 device="cpu"
                 ):

        if tokenizer != "bert-base-cased" and tokenizer != "gpt2":
            raise Warning(
                "The tokenizer is bert-base-cased, using a different tokenizer may cause problems.\n Read the TODO in the Tokenizer class in src/data_loader.py")

        self.tokenizer_name = tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.vocab_size = self.tokenizer.vocab_size
        self.max_length = max_length
        self.device = device

    @torch.no_grad()
    def __call__(self, text):
        if self.tokenizer_name == "bert-base-cased":
            return self.bert_call(text)
        if self.tokenizer_name == "gpt2":
            return self.gpt2_call(text).to(self.device)

        raise NotImplementedError("The tokenizer is not implemented")

    def bert_call(self, list_text):
        """This function is complete garbage, it should be completely rewritten
        the sole reason as to why this function exists is becouse tokenizers have a max input length
        
        args:
            list_text (str or list of str): the text to be tokenized

        returns:
            list of torch.tensor: the tokenized text
        """
        if type(list_text) != str and type(list_text) != list:
            raise TypeError("The input must be a string or a list of strings")

        if type(list_text) == str:
            list_text = [list_text]

        out = []
        for text in list_text:
            # TODO: find a way to get the token id of the [CLS] token
            tokens = [101]
            cut = self.max_length

            while len(text) > cut:
                text_piece = text[cut-self.max_length:cut]
                tokens += self.tokenizer(text_piece).input_ids[1:-1]
                cut += self.max_length

            text_piece = text[cut-self.max_length:]
            tokens += self.tokenizer(text_piece).input_ids[1:]
            out.append(torch.tensor(
                tokens, dtype=torch.long, device=self.device))

        if len(out) == 1:
            return out[0]

        return out

    def gpt2_call(self, list_text):

        assert type(list_text) == str, "The input must be a string"

        return self.tokenizer.encode(list_text, return_tensors="pt").to(self.device).view(-1)
    

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)


class CharTokenizer(Tokenizer):
    """Char level tokenizer, code adapted from https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare_char/prepare.py"""
    def __init__(self,
                 data_file,
                 device="cpu"
                 ):
        """Given a single data file, create a tokenizer that can be used to tokenize the data

            args:
                data_file (str): the path to the data file
                device (str|torch.device): the device to use for the tokenizer
        """


        self.device = device


        with open(data_file, 'r') as f:
            data = f.read()
        print(f"length of dataset in characters: {len(data):,}")

        # get all the unique characters that occur in this text
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)
        self.unique_chars=''.join(chars)

        # create a mapping from characters to integers
        self.char_to_int = {ch: i for i, ch in enumerate(chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(chars)}


    def __call__(self, input_string):
        # encoder: take a string, output a list of integers
        return torch.tensor([self.char_to_int[c] for c in input_string], dtype=torch.long, device=self.device)


    def decode(self, token_ids):
        # decoder: take a list of integers, output a string
        if type(token_ids) == int:
            return self.int_to_char[token_ids]
        token_ids=token_ids.tolist()
        if type(token_ids) == int:
            return self.int_to_char[token_ids]
        
        if type(token_ids) == torch.Tensor:
            token_ids = token_ids.tolist()
        return ''.join([self.int_to_char[i] for i in token_ids])
