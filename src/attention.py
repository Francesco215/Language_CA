import torch
import einops

from math import sqrt
from torch.autograd.function import once_differentiable


class AttentionMessageFunction(torch.autograd.Function):
    """This class implements the attention message function.
    It is a torch.autograd.Function that is used to calculate the attention message and its gradient.
    """

    @staticmethod
    def forward(ctx,
                Q: torch.Tensor,
                K: torch.Tensor,
                V: torch.Tensor,
                edge_index: torch.Tensor,
                split_size=2**15
                ):
        """This function calculates the attention message for each node in the graph.

        Args:
            K (torch.Tensor): Key tensor of shape (h, N, dK)
            Q (torch.Tensor): Query tensor of shape (h, N, dK)
            V (torch.Tensor): Value tensor of shape (h, N, dV)
            edge_index (torch.Tensor): Adjacency matrix of the graph of shape (2, M)

        Returns:
            torch.Tensor: Multi-head attention message of shape (h, N, dV)
        """
        assert K.dim() == Q.dim() == V.dim() == 3, "K, Q, V must be tensors of rank 3"
        assert K.shape[:-1] == Q.shape[:-1] == V.shape[:-1], " all dimentions of K, Q, V except the last must be the same"
        assert K.shape[-1] == Q.shape[-1], "K and Q must have the same last dimension"

        assert edge_index.dim() == 2, "edge_index must be a 2-dimentional tensor"
        assert edge_index.shape[0] == 2, "edge_index must have 2 rows"
        assert edge_index.dtype == torch.long, "edge_index must be a long tensor"

        senders, receivers = edge_index
        n_nodes, heads, d = K.shape

        # Q.K^T / sqrt(d)
        att=overlaps(Q, K, edge_index, split_size)/sqrt(d)
        
        # softmax
        attention = softmax(att, receivers, n_nodes, heads)

        # softmax*V
        out = mult_att(attention, V, senders, receivers, split_size)
       
        # save the tensors for the backward
        split_size=torch.tensor([split_size])
        ctx.save_for_backward(out,attention, Q, K, V, edge_index, split_size)

        return out, attention

    #@once_differentiable 
    @staticmethod
    def backward(ctx, grad_out, grad_attention):
        """This function calculates the gradient of the attention message function.

        Args:
            ctx (torch.autograd.function._ContextMethodMixin): Context of the backward pass
            grad_out (torch.Tensor): Gradient of the output of the attention message function
            grad_attention (torch.Tensor): Gradient of the attention tensor

        Returns:
            tuple: Tuple of gradients of the input tensors
        """
        # set the gradients of the output tensors to None    
        grad_Q = grad_K = grad_V = None

        # if the gradient of the input tensors is needed
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            out, attention, Q, K, V, edge_index, split_size = ctx.saved_tensors

            senders, receivers = edge_index
            n_nodes, heads, d = K.shape

            split_size=split_size.item()

        # calculate variables used in the gradient calculation
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            att_overlap=attention*overlaps(grad_out, V, edge_index, split_size)
            out_grad_overlap = einops.einsum(out, grad_out, '... i, ... i -> ...')

        # calculate the gradients
        if ctx.needs_input_grad[0]:
            grad_Q = compute_grad_Q(attention, K, att_overlap, out_grad_overlap, senders, receivers, split_size)/sqrt(d)
        if ctx.needs_input_grad[1]:
            grad_K = compute_grad_K(attention, Q, att_overlap, out_grad_overlap, senders, receivers, split_size)/sqrt(d)
        if ctx.needs_input_grad[2]:
            grad_V = compute_grad_V(grad_out, attention, senders, receivers, split_size)

        # return the gradients
        return grad_Q, grad_K, grad_V, None, None





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
    translation.scatter_reduce(0, receivers.repeat(heads, 1).t(), att, reduce='amax',include_self=False)

    # subtract the maximum value from the attention of each edge going to that node
    att = att-translation[receivers]

    # exponentiate the attention
    # could be done in-plase using the function att.exp_() if memory is a bootleneck
    att = torch.exp(att)

    # and normalize it to get the softmax
    return normalize_strength(att, receivers, n_nodes, heads)


def normalize_strength(strength, receivers, n_nodes, heads):
    """ This function normalizes the strength of each connection by dividing it by the sum of the
    strengths of all the connections that are directed towards the same node.

    So, lets say we have a directed graph with N nodes and M edges.
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
    assert type(n_nodes) == type(
        heads) == int, "n_nodes and heads must be integers"

    strengths_sum = torch.zeros([n_nodes, heads], device=strength.device)
    strengths_sum = strengths_sum.index_add(0, receivers, strength)

    strength = strength / strengths_sum[receivers]

    strength[strength.isnan()] = 0

    return strength



def mult_att(attention, V, senders, receivers, split_size=2**15):
    """ This function calculates the multiplication between the attention and the values.

    Args:
        attention (torch.Tensor): The attention of shape (M, h)
        V (torch.Tensor): The values of shape (N, h, dV)
        senders (torch.Tensor): The senders of shape (M,)
        receivers (torch.Tensor): The receivers of shape (M,)
        split_size (int, optional): The size of the split. Defaults to 2**15.

    Returns:
        torch.Tensor: The multiplication of the attention and the values of shape (N, h, dV)
    """

    out = torch.zeros_like(V, device=V.device)

    # split the tensors to avoid memory issues
    # in an ideal world with infinite memory we would just write this:
    # attention = einops.einsum(attention, V[senders], ' ... , ... c -> ... c')
    # out=out.index_add(0,receivers,attention)
    # return out

    for s, r, a in zip(senders.split(split_size), receivers.split(split_size), attention.split(split_size)):
        att = einops.einsum(a, V[s], ' ... , ... c -> ... c')
        # could be done in-place using the function out.index_add_()
        out = out.index_add(0, r, att)

    return out


def overlaps(Q, K, edge_index, split_size=2**15):
    """
    This function calculates the overlaps between the queries and the keys.

    Args:
        Q (torch.Tensor): Query tensor of shape (N, h, dK)
        K (torch.Tensor): Key tensor of shape (N, h, dK)
        edge_index (torch.Tensor): Adjacency matrix of the graph of shape (2, M)
        split_size (int, optional): The size of the split. Defaults to 2**15.

    Returns:
        torch.Tensor: The overlaps of shape (M, h)
    """
    senders, receivers = edge_index
    n_nodes, heads, d = K.shape

    # split the tensors to avoid memory issues
    # in an ideal world with infinite memory we would just write this:
    # return (Q[receivers]*K[senders]).sum(dim=-1)/sqrt(d)

    att = []
    for s, r in zip(senders.split(split_size), receivers.split(split_size)):
        att.append(einops.einsum(Q[r], K[s], '... c, ... c -> ...'))

    att = torch.cat(att, dim=0)

    return att






def compute_grad_Q(attention, K, att_overlap, out_grad_overlap, senders, receivers, split_size):
    """ This function calculates the gradient of the output with respect to the queries.

    Args:
        attention (torch.Tensor): The attention of shape (M, h)
        K (torch.Tensor): The keys of shape (N, h, dK)
        att_overlap (torch.Tensor): Variable calculated beforehand (M, h)
        out_grad_overlap (torch.Tensor): Variable calculated beforehand (M, h)
        senders (torch.Tensor): The senders of shape (M,)
        receivers (torch.Tensor): The receivers of shape (M,)
        split_size (int, optional): The size of the split. Defaults to 2**15.

    Returns:
        torch.Tensor: The gradient of the output with respect to the queries of shape (N, h, dK)
    """

    out = mult_att(att_overlap, K, senders, receivers, split_size)
    
    K = mult_att(attention, K, senders, receivers, split_size)
    out = out - einops.einsum(out_grad_overlap, K, '... , ... c -> ... c')

    return  out



def compute_grad_K(attention, Q, att_overlap, out_grad_overlap, senders, receivers, split_size):
    """ This function calculates the gradient of the output with respect to the keys.

    Args:
        attention (torch.Tensor): The attention of shape (M, h)
        Q (torch.Tensor): The queries of shape (N, h, dQ)
        att_overlap (torch.Tensor): Variable calculated beforehand (M, h)
        out_grad_overlap (torch.Tensor): Variable calculated beforehand (M, h)
        senders (torch.Tensor): The senders of shape (M,)
        receivers (torch.Tensor): The receivers of shape (M,)
        split_size (int, optional): The size of the split. Defaults to 2**15.

    Returns:
        torch.Tensor: The gradient of the output with respect to the keys of shape (N, h, dK)
    """
    
    out = mult_att(att_overlap, Q, receivers, senders, split_size)

    Q = einops.einsum(out_grad_overlap, Q, ' ... , ... c -> ... c')
    out = out - mult_att(attention, Q , receivers, senders, split_size)

    return out



def compute_grad_V(grad_out, attention, senders, receivers, split_size):
    """ This function calculates the gradient of the output with respect to the values.

    Args:
        grad_out (torch.Tensor): The gradient of the output of shape (N, h, dV)
        attention (torch.Tensor): The attention of shape (M, h)
        senders (torch.Tensor): The senders of shape (M,)
        receivers (torch.Tensor): The receivers of shape (M,)
        split_size (int, optional): The size of the split. Defaults to 2**15.

    Returns:
        torch.Tensor: The gradient of the output with respect to the values of shape (N, h, dV)
    """
    
    return mult_att(attention, grad_out, receivers, senders, split_size)



class AttentionMessage:
    def __init__(self, split_size=2**12):
        self.split_size = split_size
        self.function=AttentionMessageFunction.apply

    def __call__(self, Q, K, V, edge_index):
        return self.function(Q, K, V, edge_index, self.split_size)