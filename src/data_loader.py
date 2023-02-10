from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch



class Wiki(Dataset):
    def __init__(self, dataset, transform = None):
        """Creates a dataset class from a the hugging cast dataset of wikipedia 
            (https://huggingface.co/datasets/wikipedia)

        Args:
            dataset (huggingface dataset): the dataset to be used
            transform (function, optional): a function to be applied to the text, usually a tokenizer.
                Defaults to None.
        """        
        self.dataset=dataset
        self.transform=transform

    def __len__(self):
        return self.dataset.nom_rows

    def __getitem__(self,idx):
        if self.transform == None:
            return self.datatet[idx]['text']
        return self.transform(self.dataset[idx]['text'])

    def get_element(self,idx):
        return self.dataset[idx]
    
    def get_text(self,idx):
        return self.datatet[idx]['text']


class Tokenizer:

    def __init__(self,
                 tokenizer="bert-base-cased",
                 max_length=512
        ):

        if tokenizer!="bert-base-cased":
            raise Warning("The tokenizer is bert-base-cased, using a different tokenizer may cause problems.\n Read the TODO in the Tokenizer class in src/data_loader.py")

        self.tokenizer=AutoTokenizer.from_pretrained(tokenizer)
        self.vocab_size=self.tokenizer.vocab_size
        self.max_length=max_length
    
        
    def __call__(self, text):
        out=[101] #TODO: find a way to get the token id of the [CLS] token
        cut=self.max_length

        while len(text)>cut:
            text_piece=text[cut-self.max_length:cut]

            out += self.tokenizer(text_piece).input_ids[1:-1]
            cut += self.max_length
        
        text_piece=text[cut-self.max_length:]
        out+=self.tokenizer(text_piece).input_ids[1:]
        
        return torch.tensor(out)

        