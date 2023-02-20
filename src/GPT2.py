
import torch
from torch import nn

from transformers import GPT2Tokenizer

class GPT2(nn.Module):
    
    def __init__(self,
                 tokenizer=GPT2Tokenizer.from_pretrained('gpt2'),
                 n_blocks=12,
                 d_Embedding=768,
                 dK=768,
                 dV=768,
                 heads=8,
                 intermediate_size=3072,
                 dropout=0.1,
                 max_position_encoding=1024,
                 embedding_dropout=0.1,
                 device='cpu'
                 ):
        super().__init__()
        
        # Save the parameters
        self.tokenizer=tokenizer
        self.n_blocks=n_blocks
        self.d_Embedding=d_Embedding
        self.dK=dK
        self.dV=dV
        self.heads=heads
        self.intermediate_size=intermediate_size
        self.dropout=dropout,
        self.max_position_encoding=max_position_encoding,
        self.embedding_dropout=embedding_dropout,
        self.device=device

        # Initialize the embedding layer
        self.embedding=nn.Embedding(tokenizer.vocab_size, d_Embedding, device=device)
        self.positional_encoding=nn.Embedding(max_position_encoding, d_Embedding, device=device)
        self.embedding_dropout=nn.Dropout(dropout)

        # Initialize the transformer blocks
        self.blocks=nn.ModuleList([GPT2_BLock(d_Embedding, dK, dV, heads, intermediate_size, dropout, device) for _ in range(n_blocks)])
        

        # Initialize the language model head
        self.layer_norm=nn.LayerNorm(d_Embedding, eps = 1e-5, elementwise_affine=True, device=device)
        self.language_model_head=nn.Linear(d_Embedding, tokenizer.vocab_size, bias=False, device=device)
        self.activation_head=nn.Softmax(dim=-1)

        # Calculate the number of parameters
        self.n_parameters=self.blocks[0].n_parameters*n_blocks + 2*d_Embedding*tokenizer.vocab_size + max_position_encoding*d_Embedding

    def forward(self, x, edge_index):
        #Embedding
        indices=torch.arange(x.shape[0], device=self.device)
        x = self.embedding(x) + self.positional_encoding(indices)
        x=self.embedding_dropout(x)
        
        #Blocks
        for block in self.blocks:
            x=block(x, edge_index)

        #Language model head
        x=self.layer_norm(x)
        x=self.language_model_head(x)
        x=self.activation_head(x)
        return x

    def load_from_original(self, weight_token_embedding, weight_positional_embedding, transformer_heads, language_model_head):
        # Load the embedding layer
        self.embedding.weight=weight_token_embedding.weight
        self.embedding.bias=weight_token_embedding.bias
        self.positional_encoding.weight=weight_positional_embedding.weight

        # Load the transformer blocks
        for i in range(self.n_blocks):
            self.blocks[i].load_from_original(transformer_heads[i])

        # Load the language model head
        self.language_model_head.weight=language_model_head.weight
        self.language_model_head.bias=language_model_head.bias



class GPT2_BLock(nn.Module):

    def __init__(self, d_Embedding=768, dK=768, dV=768, heads=8,intermediate_size=3072, dropout=0.1, device='cpu'):
        super().__init__()

        # Save the parameters
        self.d_Embedding=d_Embedding
        self.dK=dK
        self.dV=dV
        self.heads=heads
        self.intermediate_size=intermediate_size
        self.dropout=dropout
        self.device=device

        # Initialize the transformer block and the MLP
        self.attention_block=AttentionBlockGPT2(d_Embedding, dK, dV, heads, dropout, device)
        self.MLP=GPT2MLP(d_Embedding, intermediate_size, dropout, device)

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(self.d_Embedding, eps = 1e-5, elementwise_affine=True, device=device)
        self.layer_norm2 = nn.LayerNorm(self.d_Embedding, eps = 1e-5, elementwise_affine=True, device=device)

        # Calculate the number of parameters
        self.n_parameters=self.attention_block.n_parameters + self.MLP.n_parameters

    def forward(self, x, edge_index):
        #✔
        #Attention
        residual=x
        x=self.layer_norm1(x)
        x=residual + self.attention_block(x, edge_index)
        
        #MLP
        residual=x
        x=self.layer_norm2(x)
        x= residual + self.MLP(x)
        return x
    
    def load_from_original(self, head):
        # Load the transformer block
        self.attention_block.load_from_original(head.attn.c_attn,head.attn.c_proj)

        # Load the MLP
        self.MLP.load_from_original(head.mlp.c_fc, head.mlp.c_proj)
        


from .transformerMP import AttentionBlock
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


    def load_from_original(self, c_attn, c_proj):
        w, b = c_attn.weight, c_attn.bias

        query_weight, key_weight, value_weight = torch.split(w, w.shape[-1]//3, dim=-1)
        query_bias,   key_bias,   value_bias   = torch.split(b, b.shape[-1]//3, dim=-1)

        self.key.weight=key_weight
        self.key.bias=key_bias

        self.query.weight=query_weight
        self.query.bias=query_bias

        self.value.weight=value_weight
        self.value.bias=value_bias

        self.feedforward.weight=c_proj.weight
        self.feedforward.bias=c_proj.bias



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
    def __init__(self, d_Embedding=768, intermediate_size=3072, dropout=0.1, device='cpu'):
        super().__init__()
        
        self.feedforward1=nn.Linear(d_Embedding, intermediate_size, device=device)
        self.feedforward2=nn.Linear(intermediate_size, d_Embedding, device=device)

        self.dropout=nn.Dropout(dropout, device=device)
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