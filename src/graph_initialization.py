import torch, torch_geometric
from torch_geometric.data import Data,Batch
from transformers import AutoTokenizer

def sequence_to_linear_graph(sequence:torch.Tensor):
    """Converts a sequence of nodes to a linear graph.

    Args:
        sequence (torch.Tensor): A tensor of shape (N, F) where N is the number of nodes and F is the number of features.

    Returns:
        torch_geometric.data.Data: A torch_geometric.data.Data object representing the graph.
    """
    #assert sequence.shape.shape==1, "sequence must be a 1D tensor."
    
    forward=[[i, i + 1] for i in range(sequence.shape[0] - 1)]
    backward=[[i + 1, i] for i in range(sequence.shape[0] - 1)]
    edges=forward+backward
    
    edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    return Data(x=sequence, edge_index=edges)



def sequence_to_random_graph(sequence:torch.Tensor, avg_n_edges:int=5):
    """Converts a sequence of nodes to a random graph.
    The graph is completely random, meaning that the probability of an edge
    between two nodes is the same for all nodes.

    Args:
        sequence (torch.Tensor): A tensor of shape (N, F) where N is the number of nodes and F is the number of features.
        avg_n_edges (int, optional): The average number of edges per node. Defaults to 5.

    Returns:
        torch_geometric.data.Data: A torch_geometric.data.Data object representing the graph.
    """
    #this part links successive nodes
    #assert sequence.shape.shape==1, "sequence must be a 1D tensor."

    forward=[[i, i + 1] for i in range(sequence.shape[0] - 1)]
    backward=[[i + 1, i] for i in range(sequence.shape[0] - 1)]
    edges=forward+backward
    
    edges = torch.tensor(edges, dtype=torch.long)

    #this part makes the random edges    
    rand_edges=torch.randint(0,sequence.shape[0],(avg_n_edges*sequence.shape[0], 2))
    rand_edges=remove_duplicates(rand_edges)

    edges=torch.cat((edges,rand_edges),dim=0).t().contiguous()

    return Data(x=sequence, edge_index=edges)




@torch.no_grad()
class text_to_graph():
    """Given a list of texts, returns a torch_geometric.data.Batch object containing the graphs.
    """
    def __init__(self,
                tokenizer=AutoTokenizer.from_pretrained("bert-base-cased"),
                seq_to_graph=sequence_to_random_graph):
        """
        Args:
            tokenizer (optional): A tokenizer object. 
                Defaults to AutoTokenizer.from_pretrained("bert-base-cased").
            seq_to_graph (optional): A function that converts a sequence of nodes to a graph.
                Defaults to sequence_to_random_graph.
        """     
        self.tokenizer=tokenizer
        self.vocab_size=tokenizer.vocab_size
        
        self.seq_to_graph=seq_to_graph
    
    def __call__(self,text:list)->torch_geometric.data.Batch:
        """Given a list of texts, returns a torch_geometric.data.Batch object containing the graphs.
        Args:
            text (list): A list of texts.
        Returns:
            torch_geometric.data.Batch: A torch_geometric.data.Batch object containing the graphs.
        """
        
        tokenize_texts=self.tokenizer(text)['input_ids']

        graphs = [self.seq_to_graph(torch.LongTensor(tokens)) for tokens in tokenize_texts]
        
        return Batch.from_data_list(graphs)






#utils
def remove_duplicates(edges:torch.Tensor)->torch.Tensor:
    """Removes duplicate edges from a list of edges.

    Args:
        edges (torch.Tensor): A tensor of shape (2, N) where N is the number of edges.

    Returns:
        torch.Tensor: A tensor of shape (2, M) where M is the number of unique edges.
    """
    edges = edges[edges[:,0]!=edges[:,1]]
    return torch.unique(edges,dim=0)