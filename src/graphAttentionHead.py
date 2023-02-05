from torch import nn
from torch_geometric.nn import MessagePassing
from math import sqrt

class TransformerMessagePassing (MessagePassing):
    def __init__(self, d_Embedding, dK, dV, heads=8, dropout=0.6):
        super().__init__(aggr='add')

        # Save the parameters
        self.d_Embedding=d_Embedding
        self.dK=dK
        self.dV=dV
        self.heads=heads
        self.dropout=dropout #dropout not implemented yet

        # Create the layers for the attention that make the keys, queries and values for each head
        self.key   = make_heads(d_Embedding, dK, heads)
        self.query = make_heads(d_Embedding, dK, heads)
        self.value = make_heads(d_Embedding, dV, heads)

        # Create the layer that aggregates the heads and outputs the final embedding with a linear layer
        self.out=aggregate_heads(dV, d_Embedding, heads)

        # Non-linear functions
        self.softmax=nn.Softmax(dim=-1)
        self.norm=sqrt(dK)

    def forward(self, x, edge_index):
        #calculate keys, values and queries for each head
        K=self.key(x)
        V=self.value(x)
        Q=self.query(x)

        #calculate the message and update the nodes
        return self.propagate(edge_index, x=x, K=K, Q=Q, V=V)


    def message(self,K_j,Q_i,V_j):

        #compute the attention matrix
        alpha=(Q_i*K_j).sum(dim=-1)
        alpha=alpha/self.norm
        alpha=self.softmax(alpha)

        #multiply the attention matrix with the values
        return alpha*V_j #this is the output for each head (..., heads, dV)

    def update(self, aggr_out):

        #concatenate the heads and apply the final linear layer
        return self.out(aggr_out) #(..., d_Embedding)




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