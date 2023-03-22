from torch.utils.data import Dataset
import numpy as np
import torch
from src.GPT2 import GPT2


from src.graph_initialization import Graph_maker, batch_graphs
class Wiki(Dataset):
    def __init__(self, dataset, tokenizer, graph_maker, transform = None, min_len_input=2e3, overflow_len=6000, device="cpu"):
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
        self.min_len_input=min_len_input
        self.overflow_len=overflow_len
        self.device=device

        self.index=0
        self.indices=torch.randperm(len(self.dataset))

    def __len__(self):
        return len(self.dataset)
    
    def take_text_in_order(self):
        current_len = 0
        nodes = []
        edge_index = []
        targets = []

        while(current_len <= self.min_len_input and self.index < len(self.dataset)):
            index=self.indices[self.index].item()
            text = self.dataset[index]['text']
            text = self.tokenizer(text).to(self.device)

            n = text[:-1]
            t = text[1:]

            if current_len + n.shape[0] > self.overflow_len:
                cutoff = self.overflow_len - current_len - n.shape[0]
                n = n[:cutoff]
                t = t[:cutoff]

            current_len += n.shape[0]

            nodes.append(n)
            targets.append(t)
            edge_index.append(self.graph_maker(n.shape[0]))

            self.index += 1


        nodes, edge_index = batch_graphs(nodes, edge_index)
        targets = torch.cat(targets, dim=0)

        return nodes, edge_index, targets

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


@torch.no_grad()
def validation(validation_set, model, loss_function, graph_maker, n_samples=30, ramdom=True, starting_index=0):
    """ This function is used to evaluate the model on the validation set.

    Args:
        validation_set (huggingface dataset): the validation set
        model (torch.nn.Module): the model to be evaluated
        loss_function (torch.nn.Module): the loss function to be used
        graph_maker (Graph_maker): the graph maker to be used
        n_samples (int, optional): the number of samples to be used. Defaults to 30.
        ramdom (bool, optional): if True, the samples are chosen randomly, else they are taken sequentially from the starting index.
            Defaults to True.
        starting_index (int, optional): the starting index of the samples. Defaults to 0.
     
    Returns:
        torch.tensor: the average loss on the validation set
    """

    loss = torch.tensor([0.], device=model.device)
    for i in range(starting_index, starting_index+n_samples):

        if ramdom:
            i = np.random.randint(0, len(validation_set))

        text = validation_set[i]['text']
        x = model.tokenizer(text)
        if isinstance(model, GPT2):
            x = x[:1023]

        nodes = x[:-1]
        target = x[1:]

        edge_index = graph_maker(nodes.shape[0])

        out = model(nodes, edge_index)
        loss += loss_function(out, target)

    return loss/n_samples

