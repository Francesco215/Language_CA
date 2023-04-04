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
        self.make_QKV = make_QKV(d_Embedding, dK, dV, heads, rotary_encoding, True, device)

        # Create the attention layer
        self.attention_message = AttentionMessage(split_size)

        # Create the layer that aggregates the heads and outputs the final embedding with a linear layer
        self.feedforward = aggregate_heads(dV, d_Embedding, heads, device)

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        # Calculate the number of parameters
        self.n_parameters = self.make_QKV.n_parameters + self.feedforward.n_parameters

    def forward(self, x, edge_index):
        """This function calculates the messages for each node in the graph.

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

        return x


from src.positional_encoding import RotaryEncoding
# Utilis function to make the code more readable. they are just to make the generation of K,Q,V
# with multi-head and going back to the embedding much easier to read
class make_QKV(nn.Linear):
    def __init__(self, d_Embedding, dK, dV, heads, rotary_encoding=False, bias=True, device='cpu'):
        """ This class is a linear layer that splits the output in three parts:
        Q, K and V. The output is split in three parts of size dK*heads, dK*heads and dV*heads.

        Args:
            d_Embedding (int): The dimension of the input
            dK (int): The dimension of the queries and keys
            dV (int): The dimension of the values
            heads (int): The number of heads
            rotary_encoding (bool, optional): If true, rotary encoding in applied to the queries and keys.
                Defaults to False.
            bias (bool, optional): If true, a bias is added to the output. Defaults to True.
            device (str, optional): The device where the layer is located. Defaults to 'cpu'.
        """
        out_features = (2*dK+dV)*heads
        super().__init__(d_Embedding, out_features, bias, device=device)

        self.d_Embedding = d_Embedding
        self.dK = dK
        self.dV = dV
        self.heads = heads
        self.rotary_encoding = rotary_encoding
        if rotary_encoding:
            self.rotary_encoding = RotaryEncoding()

        self.split_shape = (dK*heads, dK*heads, dV*heads)

        self.n_parameters = self.in_features*self.out_features

    def forward(self, x):
        """
            This function splits the output of the linear layer in three parts:
            Q, K and V. The output is split in three parts of size dK*heads, dK*heads and dV*heads.

            Args:
                x (torch.Tensor): The input of the layer of shape (..., sequence_length, d_Embedding)

            Returns:
                torch.Tensor: The queries of shape (..., heads, sequence_length, dK)
                torch.Tensor: The keys of shape (..., heads, sequence_length, dK)
                torch.Tensor: The values of shape (..., heads, sequence_length, dV)
        """

        # split the output in three parts
        Q, K, V = super().forward(x).split(self.split_shape, dim=-1)

        # reshape the output to have the correct shape
        Q = Q.view(-1, self.heads, self.dK)
        K = K.view(-1, self.heads, self.dK)
        V = V.view(-1, self.heads, self.dV)

        # apply rotary encoding if needed
        if isinstance(self.rotary_encoding, RotaryEncoding):
            Q = self.rotary_encoding(Q)
            K = self.rotary_encoding(K)

        return Q, K, V


class aggregate_heads(nn.Linear):

    def __init__(self, dV, d_Embedding, heads, device='cpu'):
        super().__init__(dV*heads, d_Embedding, device=device)

        self.n_parameters = self.in_features*self.out_features

    def forward(self, x):
        return super().forward(x.view(-1, self.in_features))



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
