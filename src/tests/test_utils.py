import torch

from src.attention import overlaps, softmax, mult_att
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
    It's hard to read, but it's the only way I found to make it fast and parallelizable.
    Args:
        K (torch.Tensor): Key tensor of shape (N, h, dK)
        Q (torch.Tensor): Query tensor of shape (N, h, dK)
        V (torch.Tensor): Value tensor of shape (N, h, dV)
        edge_index (torch.Tensor): Adjacency matrix of the graph of shape (2, M)
    Returns:
        torch.Tensor: Multi-head attention message of shape (N, h, dV)
    """
    assert K.dim() == Q.dim() == V.dim() == 3, "K, Q, V must be tensors of rank 3"
    assert K.shape[0] == Q.shape[0] == V.shape[0], "K, Q, V must have the same first dimension"
    assert K.shape[1] == Q.shape[1] == V.shape[1], "K, Q, V must have the same second dimension"
    assert K.shape[2] == Q.shape[2], "K and Q must have the same third dimension"

    assert edge_index.dim() == 2, "edge_index must be a 2-dimentional tensor"
    assert edge_index.shape[0] == 2, "edge_index must have 2 rows"
    assert edge_index.dtype == torch.long, "edge_index must be a long tensor"

    senders, receivers = edge_index
    n_nodes, heads, d = K.shape

    # Q.K^T / sqrt(d)
    att=overlaps(Q, K, edge_index, split_size)
    
    # softmax
    attention = softmax(att, receivers, n_nodes, heads)

    # softmax*V
    out = mult_att(attention, V, senders, receivers, split_size)
    
    # save the tensors for the backward
    split_size=torch.tensor([split_size])

    return out, attention
