from torch import nn
import torch
from math import sqrt
import einops


class AttentionBlock(nn.Module):
    """
    This class is a message passing layer that uses the transformer architecture to calculate the messages.
    The transformer architecture is based on the paper "Attention is all you need" by Vaswani et al. (2017).
    """

    def __init__(self, d_Embedding=512, dK=1024, dV=1024, heads=8, dropout=0.0, device='cpu'):

        super().__init__()

        # Save the parameters
        self.d_Embedding = d_Embedding
        self.dK = dK
        self.dV = dV
        self.dQ = dK
        self.heads = heads
        self.dropout = dropout
        self.device = device

        # Create the layers for the attention that make the keys, queries and values for each head
        self.make_QKV = make_QKV(d_Embedding, dK, dV, heads, device)

        # Create the layer that aggregates the heads and outputs the final embedding with a linear layer
        self.feedforward = aggregate_heads(dV, d_Embedding, heads, device)

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        # Calculate the number of parameters
        self.n_parameters = self.make_QKV.n_parameters + self.feedforward.n_parameters

    def forward(self, x, edge_index):
        """Here the forward is different than in the figure 1 of attention is all you need.
           The output of the attention is summed to the input x only at the end of the forward and
           there is no normalization of the output.
           There are a few reason to do so:
              1) When the system reaches stability, we want the output of the transformer block to be zero (or close to zero)
              2) We don't want the positioning encoding to go away. If we normalize the output of the attention, we will lose the positional encoding.
              3) The gradient will propagate easly

        Args:
            x (torch.Tensor): The values of the nodes of the graph
            edge_index (torch.Tensor): adjacency matrix of the graph

        Returns:
            torch.Tensor: Updated values of the nodes of the graph
        """

        # calculate keys, values and queries for each head
        # This can be optimized by doing something like this:
        Q, K, V = self.make_QKV(x)

        # calculate the multi-head attention and activation function
        # TODO: check if residual should be added before or after the linear layer or at all
        x, _ = attention_message(Q, K, V, edge_index, self.dropout)

        # now we merge the output of the heads and apply the final linear layer
        x = self.feedforward(x)
        x = self.dropout_layer(x)

        # There is no add and normalize here!

        return x


def attention_message(Q: torch.Tensor,
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
    assert K.dim() == Q.dim() == V.dim() == 3, "K,Q,V must be 3-dimentional tensors"
    assert K.shape[0] == Q.shape[0] == V.shape[0], "K,Q,V must have the same first dimension"
    assert K.shape[1] == Q.shape[1] == V.shape[1], "K,Q,V must have the same second dimension"
    assert K.shape[2] == Q.shape[2], "K,Q must have the same third dimension"

    assert edge_index.dim() == 2, "edge_index must be a 2-dimentional tensor"
    assert edge_index.shape[0] == 2, "edge_index must have 2 rows"
    assert edge_index.dtype == torch.long, "edge_index must be a long tensor"

    senders, receivers = edge_index
    n_nodes, heads, d = K.shape

    # Q.K^T this is the line of code that uses a lot of memory
    att = []
    for s, r in zip(senders.split(split_size), receivers.split(split_size)):
        att.append((Q[r]*K[s]).sum(dim=-1)/sqrt(d))

    att = torch.cat(att, dim=0)

    # softmax
    attention = softmax(att, receivers, n_nodes, heads)

    # softmax*V
    out = torch.zeros_like(V, device=V.device)
    for s, r, a in zip(senders.split(split_size), receivers.split(split_size), attention.split(split_size)):
        att = einops.einsum(a, V[s], ' ... , ... c -> ... c')
        # could be done in-place using the function out.index_add_()
        out = out.index_add(0, r, att)

    return out, attention


def softmax(att, receivers, n_nodes, heads):
    """
    This function calculates the softmax of the attention.

    Args:
        att (torch.Tensor): The attention of shape (M, h)
        receivers (torch.Tensor): The receivers of shape (M,)
        n_nodes (int): The number of nodes in the graph
        heads (int): The number of heads in the multi-head attention

        Returns:
            torch.Tensor: The softmax of the attention of shape (M, h)
    """

    # Create the translation tensor to avoid numerical instabilities
    translation = torch.zeros(n_nodes, heads, device=receivers.device)

    # take the maximum value of the attention going to each node
    translation = translation.scatter_reduce(0, receivers.repeat(heads, 1).t(), att, reduce='amax', include_self=False)
    #TODO: check if this is faster 
    #translation.scatter_reduce(0, receivers.repeat(heads, 1).t(), att, reduce='amax',include_self=False)

    # subtract the maximum value from the attention of each edge going to that node
    att = att-translation[receivers]

    # exponentiate the attention
    # could be done in-plase using the function att.exp_() if memory is a bootleneck
    att = torch.exp(att)

    # and normalize it to get the softmax
    return normalize_strength(att, receivers, n_nodes, heads)


def normalize_strength(strength, receivers, n_nodes, heads):
    """ If you think defining a whole function for 3 lines of code is overkill, try to understand it.
    lets say we have a directed graph with N nodes and M edges.
    To represent each one i have 3 M-dimentional vectors which are cal `senders`, `receivers`
    and `strength`:
    The i-th element of the `senders` vector represents a node  that is directed towards the
    i-th element of the `receivers` vector. The strength of this connection is represented by
    the i-th element of the `strength` vector.

    This function normalizes the strength of each connection by dividing it by the sum of the
    strengths of all the connections that are directed towards the same node.

    Args:
        receivers (torch.Tensor): A 1D-tensor of length n_edges
        strength (torch.Tensor): strength of each connection, (M,h) where M is the number of edges
            head is the number of heads
        n_nodes (int): number of nodes
        heads (int): number of heads

    Returns:
        torch.Tensor: strenght vector normalized by the sum of the strengths of all the
            connections that are directed towards the same node.
    """
    assert strength.dim() == 2, "strength must be a 2-dimentional tensor (M,h) where head is the number of heads"
    assert type(n_nodes) == type(heads) == int, "n_nodes and heads must be integers"

    strengths_sum = torch.zeros([n_nodes, heads], device=strength.device)
    strengths_sum = strengths_sum.index_add(0, receivers, strength)

    strength = strength / strengths_sum[receivers]

    strength[strength.isnan()] = 0

    return strength



# Utilis function to make the code more readable. they are just to make the generation of K,Q,V
# with multi-head and going back to the embedding much easier to read
class make_QKV(nn.Linear):
    def __init__(self, d_Embedding, dK, dV, heads, device='cpu'):
        out_features = (2*dK+dV)*heads
        super().__init__(d_Embedding, out_features, device=device)

        self.d_Embedding = d_Embedding
        self.dK = dK
        self.dV = dV
        self.heads = heads

        self.split_shape = (dK*heads, dK*heads, dV*heads)

        self.n_parameters = self.weight.shape[0]*self.weight.shape[1]

    def forward(self, x):

        Q, K, V = super().forward(x).split(self.split_shape, dim=-1)

        Q = Q.view(-1, self.heads, self.dK)
        K = K.view(-1, self.heads, self.dK)
        V = V.view(-1, self.heads, self.dV)

        return Q, K, V


class aggregate_heads(nn.Linear):

    def __init__(self, dK, d_Embedding, heads, device='cpu'):
        super().__init__(dK*heads, d_Embedding, device=device)

        self.x_dim = heads*dK
        self.n_parameters = self.weight.shape[0]*self.weight.shape[1]

    def forward(self, x):
        return super().forward(x.view(-1, self.x_dim))



class BlockGenerator:
    def __init__(self, block, *args, **kwargs):
        """Generates a block of the graph attention network

        Args:
            block : class initializer
        """
        #assert isinstance(block, type), "block must be a class initializer (not an instance)"
        self.block = block

        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        out = self.block(*self.args, **self.kwargs)
        assert isinstance(out, AttentionBlock), "block must be a subclass of AttentionBlock"
        return out
