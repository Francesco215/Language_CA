from torch import nn
from torch_geometric.nn import MessagePassing
from math import sqrt


class TransformerBlock(MessagePassing):
    """
    This class is a message passing layer that uses the transformer architecture to calculate the messages.
    The transformer architecture is based on the paper "Attention is all you need" by Vaswani et al. (2017).
    """
    #TODO: Implement the dropout
    def __init__(self, d_Embedding=512, dK=1024, dV=1024, heads=8):
        super().__init__(aggr='add')

        # Save the parameters
        self.d_Embedding=d_Embedding
        self.dK=dK
        self.dV=dV
        self.heads=heads

        # Create the layers for the attention that make the keys, queries and values for each head
        self.key   = make_heads(d_Embedding, dK, heads)
        self.query = make_heads(d_Embedding, dK, heads)
        self.value = make_heads(d_Embedding, dV, heads)

        # Create the layer that aggregates the heads and outputs the final embedding with a linear layer
        self.head_aggregator=aggregate_heads(dV, d_Embedding, heads)

        # Non-linear functions
        self.softmax=nn.Softmax(dim=-1)
        self.norm=sqrt(dK)

        # Activation function and feedforward layer
        self.activation=nn.ReLU()
        self.feedforward=nn.Linear(d_Embedding, d_Embedding)
        

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
        out = self.propagate(edge_index, x=x, K=K, Q=Q, V=V)
        out = self.activation(out)

        #There is no add and normalize here!

        out = self.feedforward(out)
        out = self.activation(out)

        #There is no add and normalize here!

        return x + out

    def message(self,K_j,Q_i,V_j):

        #compute the attention matrix
        alpha=(Q_i*K_j).sum(dim=-1)
        alpha=alpha/self.norm
        alpha=self.softmax(alpha)

        #multiply the attention matrix with the values
        return alpha*V_j #this is the output for each head (..., heads, dV)

    def update(self, aggr_out):

        #concatenate the heads and apply the final linear layer
        return self.head_aggregator(aggr_out) #(..., d_Embedding)




class make_heads(nn.Linear):

    def __init__(self,in_features,out_features,heads=8):
        super().__init__(in_features,out_features*heads)
        self.out_features=out_features
        self.heads=heads

    def forward(self,x):
        return super().forward(x).view(-1,self.heads,self.out_features)

class aggregate_heads(nn.Linear):
    
        def __init__(self,in_features,out_features,heads=8):
            super().__init__(in_features*heads,out_features)
            self.out_features=out_features
            self.heads=heads
    
        def forward(self,x):
            return super().forward(x.view(-1,self.heads*self.out_features))