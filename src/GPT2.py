
import torch
from torch import nn

from .transformerMP import AttentionBlock

class GPT2_BLock(nn.Module):

    def __init__(self, tranformer_block, MLP, device='cpu'):
        super().__init__()

        assert isinstance(tranformer_block, AttentionBlock), "tranformer_block must be an instance of TransformerBlock"
        assert isinstance(MLP, GPT2MLP), "MLP must be an instance of GPT2MLP"

        # Save the parameters
        self.transformer_block=tranformer_block
        self.MLP=MLP
        self.device=device
        self.d_embedding=self.transformer_block.d_Embedding

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(self.d_Embedding, eps = 1e-5, elementwise_affine=True)
        self.layer_norm2 = nn.LayerNorm(self.d_Embedding, eps = 1e-5, elementwise_affine=True)

        # Calculate the number of parameters
        self.n_parameters=self.transformer_block.n_parameters + self.MLP.n_parameters

    def forward(self,x, edge_index):
        #✔
        #Attention
        residual=x
        x=self.layer_norm1(x)
        x=residual + self.transformer_block(x, edge_index)
        
        #MLP
        residual=x
        x=self.layer_norm2(x)
        x= residual + self.MLP(x)
        return x
        



class AttentionBlockGPT2(AttentionBlock):
    """
    This class is a message passing layer that uses the transformer architecture to calculate the messages.
    The transformer architecture is based on the GPT-2 paper "Language Models are Unupervised Multitask Learners".
    """
    def __init__(self, d_Embedding=768, dK=768, dV=768, heads=8, dropout=0.1, device='cpu'):

        assert d_Embedding%heads==0, "d_Embedding must be divisible by heads"

        super().__init__(d_Embedding, dK, dV, heads, dropout, device)

        # Create the layers for the attention that make the keys, queries and values for each head
        self.key   = transform_heads(d_Embedding, dK, heads, device)
        self.query = transform_heads(d_Embedding, dK, heads, device)
        self.value = transform_heads(d_Embedding, dV, heads, device)

        # Create the layer that aggregates the heads and outputs the final embedding with a linear layer
        self.feedforward=interact_heads(dV, d_Embedding, device)

        # Calculate the number of parameters
        self.n_parameters=self.key.n_parameters*2 + self.value.n_parameters + self.feedforward.n_parameters 


    def load_from_original(self, c_attn):
        w, b = c_attn.weight, c_attn.bias

        query_weight, key_weight, value_weight = torch.split(w, w.shape[-1]//3, dim=-1)
        query_bias,   key_bias,   value_bias   = torch.split(b, b.shape[-1]//3, dim=-1)

        self.key.weight=key_weight
        self.key.bias=key_bias

        self.query.weight=query_weight
        self.query.bias=query_bias

        self.value.weight=value_weight
        self.value.bias=value_bias



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
    




class GPT2MLP(nn.Module):
    #✔
    #souce: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
    #line 334
    def __init__(self, d_Embedding=768, intermediate_size=3072, dropout=0.1):
        super().__init__()
        
        self.feedforward1=nn.Linear(d_Embedding, intermediate_size)
        self.feedforward2=nn.Linear(intermediate_size, d_Embedding)

        self.dropout=nn.Dropout(dropout)
        self.activation=NewGELUActivation()

        self.n_parameters=2*d_Embedding*intermediate_size

    def forward(self,x):
        #✔
        x=self.feedforward1(x)
        x=self.activation(x)
        x=self.feedforward2(x)
        x=self.dropout(x)

        return x


    def load_from_original(self,c_fc,c_proj):
        self.feedforward1.weight=c_fc.weight
        self.feedforward1.bias=c_fc.bias

        self.feedforward2.weight=c_proj.weight
        self.feedforward2.bias=c_proj.bias


import math
class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))