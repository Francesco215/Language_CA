from torch import nn
import torch
from math import sqrt
import einops


class TransformerBlock(nn.Module):
    """
    This class is a message passing layer that uses the transformer architecture to calculate the messages.
    The transformer architecture is based on the paper "Attention is all you need" by Vaswani et al. (2017).
    """
    #TODO: Implement the dropout
    def __init__(self, d_Embedding=512, dK=1024, dV=1024, heads=8, device='cpu'):

        super().__init__()
        # Save the parameters
        self.d_Embedding=d_Embedding
        self.dK=dK
        self.dV=dV
        self.heads=heads
        self.device=device

        # Create the layers for the attention that make the keys, queries and values for each head
        self.key   = make_heads(d_Embedding, dK, heads, device)
        self.query = make_heads(d_Embedding, dK, heads, device)
        self.value = make_heads(d_Embedding, dV, heads, device)

        # Create the layer that aggregates the heads and outputs the final embedding with a linear layer
        self.head_aggregator=aggregate_heads(dV, d_Embedding, heads, device)

        # Activation function and feedforward layer
        self.activation=nn.ReLU()
        self.feedforward=nn.Linear(d_Embedding, d_Embedding,device=device)
        
        # Calculate the number of parameters = # attention parameters + # feedforward parameters
        self.n_parameters=d_Embedding*heads*2*(dK+dV) + d_Embedding**2 

    def forward(self, x, edge_index):
        """Here the forward is different than in the figure 1 of attention is all you need.
           The output of the attention is summed to the input x only at the end of the forward and
           there is no normalization of the outpur.
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

        #calculate keys, values and queries for each head
        K = self.key(x)
        V = self.value(x)
        Q = self.query(x)

        #calculate the multi-head attention and activation function
        out = attention_message(K,Q,V,edge_index)
        out = self.activation(out)

        #now we merge the output of the heads and apply the final linear layer
        out=self.head_aggregator(out)
        out=self.activation(out)

        #There is no add and normalize here!

        out = self.feedforward(out)
        out = self.activation(out)

        #There is no add and normalize here!

        return x + out




def attention_message(K:torch.Tensor,
                      Q:torch.Tensor,
                      V:torch.Tensor,
                      edge_index:torch.Tensor
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
    assert K.dim()==Q.dim()==V.dim()==3, "K,Q,V must be 3-dimentional tensors"
    assert K.shape[0]==Q.shape[0]==V.shape[0], "K,Q,V must have the same first dimension"
    assert K.shape[1]==Q.shape[1]==V.shape[1], "K,Q,V must have the same second dimension"
    assert K.shape[2]==Q.shape[2], "K,Q must have the same third dimension"

    assert edge_index.dim()==2, "edge_index must be a 2-dimentional tensor"
    assert edge_index.shape[0]==2, "edge_index must have 2 rows"
    assert edge_index.dtype==torch.long, "edge_index must be a long tensor"

    receivers,senders=edge_index
    N,h,d=K.shape   

    #Q.K^T 
    att=(Q[receivers]*K[senders]).sum(dim=-1)/sqrt(d)

    #softmax    
    att = torch.exp(att) #could be done in-plase using the function att.exp_()
    att = normalize_strength(att, receivers, N, h)

    #softmax*V
    att = einops.einsum(att,V[senders],' ... , ... c -> ... c')
    out=torch.zeros_like(V,device=V.device)

    return out.index_add(0,receivers,att) #could be done in-place using the function out.index_add_()


def normalize_strength(strength,receivers,n_nodes,heads):
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
    assert strength.dim()==2, "strength must be a 2-dimentional tensor (M,h) where head is the number of heads"
    assert type(n_nodes)==type(heads)==int, "n_nodes and heads must be integers"

    strengths_sum = torch.zeros([n_nodes,heads],device=strength.device)
    strengths_sum.index_add_(0, receivers, strength)

    return strength / strengths_sum[receivers]



#Utilis function to make the code more readable. they are just to make the generation of K,Q,V
#with multi-head and going back to the embedding much easier to read
class make_heads(nn.Linear):

    def __init__(self, in_features, out_features, heads=8, device='cpu'):
        super().__init__(in_features, out_features*heads, device=device)
        self.out_features = out_features #this overwrites the out_features of the nn.Linear class but shouldn't matter
        self.heads = heads

    def forward(self,x):
        return super().forward(x).view(-1,self.heads,self.out_features)

class aggregate_heads(nn.Linear):
    
        def __init__(self, in_features, out_features, heads=8, device='cpu'):
            super().__init__(in_features*heads, out_features, device=device)
            self.x_dim=heads*in_features

        def forward(self,x):
            return super().forward(x.view(-1,self.x_dim))