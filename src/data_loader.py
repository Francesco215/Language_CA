from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
from .graph_initialization import *

class Wiki(Dataset):
    def __init__(self, dataset, tokenizer, graph_maker, transform = None, device="cpu"):
        """Creates a dataset class from a the hugging cast dataset of wikipedia 
            (https://huggingface.co/datasets/wikipedia)

        Args:
            dataset (huggingface dataset): the dataset to be used
            transform (function, optional): a function to be applied to the text, usually a tokenizer.
                Defaults to None.
        """   
        #parameters     
        #assert isinstance(tokenizer, Tokenizer), "tokenizer must be an instance of the Tokenizer class"
        assert isinstance(graph_maker, Graph_maker), "graph_maker must be an instance of the GraphMaker class"

        self.dataset=dataset
        self.tokenizer=tokenizer
        self.graph_maker=graph_maker
        self.transform=transform

    def __len__(self):
        return len(self.dataset)

    @torch.no_grad()
    def __getitem__(self,idx):
        dataset=self.dataset[idx]
        

        if type(idx)==int:
            tokenized_text=self.tokenizer(dataset['text'])
            graphs = self.graph_maker(tokenized_text.shape[0])
        elif type(idx)==slice:
            tokenized_text=[self.tokenizer(string['text']) for string in dataset]
            graphs = [self.graph_maker(text.shape[0]) for text in tokenized_text]
        else:
            raise TypeError("The index must be an int or a slice")

        return tokenized_text, graphs 

    def get_original_element(self,idx):
        return self.dataset[idx]
    
    def get_original_text(self,idx):
        return self.datatet[idx]['text']

class Tokenizer:

    def __init__(self,
                 tokenizer="bert-base-cased",
                 max_length=512
        ):

        if tokenizer!="bert-base-cased" and tokenizer!="gpt2":
            raise Warning("The tokenizer is bert-base-cased, using a different tokenizer may cause problems.\n Read the TODO in the Tokenizer class in src/data_loader.py")

        self.tokenizer_name=tokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(tokenizer)
        self.vocab_size=self.tokenizer.vocab_size
        self.max_length=max_length
    
    @torch.no_grad()
    def __call__(self,text):
        if self.tokenizer_name=="bert-base-cased":
            return self.bert_call(text)
        if self.tokenizer_name=="gpt2":
            return self.gpt2_call(text)

        raise NotImplementedError("The tokenizer is not implemented")

    def bert_call(self, list_text):
        """This function is complete garbage, it should be completely rewritten
        the sole reason as to why this function exists is becouse tokenizers have a max input length
        
        args:
            list_text (str or list of str): the text to be tokenized

        returns:
            list of torch.tensor: the tokenized text
        """
        if type(list_text)!=str and type(list_text)!=list:
            raise TypeError("The input must be a string or a list of strings")

        if type(list_text)==str:
            list_text=[list_text]

        out=[]
        for text in list_text:
            tokens=[101] #TODO: find a way to get the token id of the [CLS] token
            cut=self.max_length

            while len(text)>cut:
                text_piece=text[cut-self.max_length:cut]
                tokens += self.tokenizer(text_piece).input_ids[1:-1]
                cut += self.max_length
            
            text_piece=text[cut-self.max_length:]
            tokens+=self.tokenizer(text_piece).input_ids[1:]
            out.append(torch.tensor(tokens,dtype=torch.long))
        
        if len(out)==1:
            return out[0]

        return out
    
    def gpt2_call(self, list_text):

        assert type(list_text)==str, "The input must be a string"
        
        return self.tokenizer.encode(list_text, return_tensors="pt").view(-1)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)