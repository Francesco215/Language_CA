import torch
import einops
from math import sqrt
"""
this is the same as the function in src/attention.py
the only difference is that when executed the gradient is calculated
by the autodiff tool of pytorch

This way we can test that the gradient is correct
"""



def og_attention_message(Q: torch.Tensor,
                         K: torch.Tensor,
                         V: torch.Tensor,
                         edge_index: torch.Tensor,
                         att_dropout=0.0,
                         split_size=2**15
                         ):
    """This function calculates the attention message for each node in the graph.

    Args:
        K (torch.Tensor): Key tensor of shape (N, h, dK)
        Q (torch.Tensor): Query tensor of shape (N, h, dK)
        V (torch.Tensor): Value tensor of shape (N, h, dV)
        edge_index (torch.Tensor): Adjacency matrix of the graph of shape (2, M)

    Returns:
        torch.Tensor: Multi-head attention message of shape (N, h, dV)
    """
    n,h,d=Q.shape
    senders, receivers = edge_index

    attention=einops.einsum(K,Q,'n h d, m h d -> n m h')/sqrt(d)

    mask=torch.zeros((n,n,h))
    for i in range(edge_index.shape[1]):
        mask[senders[i],receivers[i],:]=1

    attention=attention.softmax(dim=0)*mask
    attention = torch.nn.functional.normalize(attention, p=1, dim=0)

    out=einops.einsum(attention,V,'m n h, m h d -> n h d')

    return out, attention

