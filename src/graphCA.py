import torch
from torch_geometric.data import Data
from .utils import remove_duplicates

def sequence_to_linear_graph(sequence:torch.Tensor):
    """Converts a sequence of nodes to a linear graph.

    Args:
        sequence (torch.Tensor): A tensor of shape (N, F) where N is the number of nodes and F is the number of features.

    Returns:
        torch_geometric.data.Data: A torch_geometric.data.Data object representing the graph.
    """
    forward=[[i, i + 1] for i in range(sequence.shape[0] - 1)]
    backward=[[i + 1, i] for i in range(sequence.shape[0] - 1)]
    edges=forward+backward

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=sequence, edge_index=edge_index)



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
    forward=[[i, i + 1] for i in range(sequence.shape[0] - 1)]
    backward=[[i + 1, i] for i in range(sequence.shape[0] - 1)]
    edges=torch.tensor(forward+backward)

    rand_edges=torch.randint(0,sequence.shape[0],(avg_n_edges*sequence.shape[0], 2))
    rand_edges=remove_duplicates(rand_edges)

    edges=torch.cat((edges,rand_edges),dim=0).t()


    return Data(x=sequence, edge_index=edges)
    

