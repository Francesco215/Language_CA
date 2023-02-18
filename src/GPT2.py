
import torch
from torch import nn

from .transformerMP import TransformerBlock

class GPT2(nn.Module):

    def __init__(self):
        super().__init__()

        pass

    def forward(self):
        pass



class TransformerBlockGPT2(TransformerBlock):
    """
    This class is a message passing layer that uses the transformer architecture to calculate the messages.
    The transformer architecture is based on the GPT-2 paper "Language Models are Unupervised Multitask Learners".
    """
    def __init__(self, d_Embedding=512, dK=1024, dV=1024, heads=8, dropout=0.1, device='cpu'):

        assert d_Embedding%heads==0, "d_Embedding must be divisible by heads"

        super().__init__(d_Embedding=512, dK=1024, dV=1024, heads=8, dropout=0.1)
        
        # Create the layers for the attention that make the keys, queries and values for each head
        self.key   = transform_heads(d_Embedding, dK, heads, device)
        self.query = transform_heads(d_Embedding, dK, heads, device)
        self.value = transform_heads(d_Embedding, dV, heads, device)

        # Create the layer that aggregates the heads and outputs the final embedding with a linear layer
        self.feedforward=interact_heads(dV, d_Embedding, device)

        # Calculate the number of parameters
        self.n_parameters=self.key.n_parameters*2 + self.value.n_parameters + self.feedforward.n_parameters 






#Utilis function to make the code more readable. they are just to make the generation of K,Q,V
#with multi-head and going back to the embedding much easier to read
class transform_heads(nn.Linear):

    def __init__(self, in_features, out_features, heads, device='cpu'):

        assert out_features%heads==0, "out_features must be divisible by heads"
        
        super().__init__(in_features, out_features, device=device)
        
        self.out_features = out_features #this overwrites the out_features of the nn.Linear class but shouldn't matter
        self.heads = heads
        self.n_parameters = self.weight.shape[0]*self.weight.shape[1]

    def forward(self,x):
        return super().forward(x).view(-1,self.heads,self.out_features//self.heads)

class interact_heads(nn.Linear):
    
    def __init__(self, in_features, out_features, device='cpu'):
        
        super().__init__(in_features, out_features, device=device)
        
        self.n_parameters=self.weight.shape[0]*self.weight.shape[1]


    def forward(self,x):
        return super().forward(x.view(-1,self.in_features))