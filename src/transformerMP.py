from torch import nn
import torch

from src.attention import AttentionMessage

class AttentionBlock(nn.Module):
    """
    This class is a message passing layer that uses the transformer architecture to calculate the messages.
    The transformer architecture is based on the paper "Attention is all you need" by Vaswani et al. (2017).
    """

    def __init__(self, d_Embedding=512, dK=1024, dV=1024, heads=8, dropout=0.0, rotary_encoding=False, device='cpu', split_size=2**12):

        super().__init__()

        # Save the parameters
        self.d_Embedding = d_Embedding
        self.dK = dK
        self.dV = dV
        self.dQ = dK
        self.heads = heads
        self.dropout = dropout
        self.rotary_embedding = rotary_encoding
        self.device = device
        self.split_size = split_size

        # Create the layers for the attention that make the keys, queries and values for each head
        self.make_QKV = make_QKV(d_Embedding, dK, dV, heads, rotary_encoding, device)

        # Create the attention layer
        self.attention_message = AttentionMessage(split_size)

        # Create the layer that aggregates the heads and outputs the final embedding with a linear layer
        self.feedforward = aggregate_heads(dV, d_Embedding, heads, device)

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        # Calculate the number of parameters
        self.n_parameters = self.make_QKV.n_parameters + self.feedforward.n_parameters

    def forward(self, x, edge_index):
        """
        This function calculates the messages for each node in the graph.

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
        x, _ = self.attention_message(Q, K, V, edge_index)

        # now we merge the output of the heads and apply the final linear layer
        x = self.feedforward(x)
        x = self.dropout_layer(x)

        # There is no add and normalize here!

        return x


from src.encoder import rotary_encoding
# Utilis function to make the code more readable. they are just to make the generation of K,Q,V
# with multi-head and going back to the embedding much easier to read
class make_QKV(nn.Linear):
    def __init__(self, d_Embedding, dK, dV, heads, rotary_encoding=False, device='cpu'):
        out_features = (2*dK+dV)*heads
        super().__init__(d_Embedding, out_features, device=device)

        self.d_Embedding = d_Embedding
        self.dK = dK
        self.dV = dV
        self.heads = heads
        self.rotary_encoding = rotary_encoding

        self.split_shape = (dK*heads, dK*heads, dV*heads)

        self.n_parameters = self.weight.shape[0]*self.weight.shape[1]

    def forward(self, x):

        Q, K, V = super().forward(x).split(self.split_shape, dim=-1)

        Q = Q.view(-1, self.heads, self.dK)
        K = K.view(-1, self.heads, self.dK)
        V = V.view(-1, self.heads, self.dV)

        if self.rotary_encoding:
            Q = rotary_encoding(Q)
            K = rotary_encoding(K)

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
