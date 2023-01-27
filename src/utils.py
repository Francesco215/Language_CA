import torch

def remove_duplicates(edges:torch.Tensor)->torch.Tensor:
    """Removes duplicate edges from a list of edges.

    Args:
        edges (torch.Tensor): A tensor of shape (2, N) where N is the number of edges.

    Returns:
        torch.Tensor: A tensor of shape (2, M) where M is the number of unique edges.
    """
    edges = edges[edges[:,0]!=edges[:,1]]
    return torch.unique(edges,dim=0)