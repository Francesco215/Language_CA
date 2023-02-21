import torch
from torch import nn

from src import Tokenizer
class GPT2(nn.Module):
    
    def __init__(self,
                 encoder,
                 decoder,
                 tokenizer=Tokenizer('gpt2'),
                 n_blocks=12,
                 d_Embedding=768,
                 dK=768,
                 dV=768,
                 heads=8,
                 intermediate_size=3072,
                 dropout=0.1,
                 device='cpu'
                 ):
        super().__init__()

        assert isinstance(encoder, GPT2_Encoder)
        assert isinstance(decoder, GPT2_LM_Head)
        assert isinstance(tokenizer, Tokenizer)

        # Save the parameters
        self.tokenizer=tokenizer
        self.encoder=encoder
        self.decoder=decoder
        self.n_blocks=n_blocks
        self.d_Embedding=d_Embedding
        self.dK=dK
        self.dV=dV
        self.heads=heads
        self.intermediate_size=intermediate_size
        self.dropout=dropout,
        self.device=device


        # Initialize the transformer blocks
        self.transformer_blocks=nn.ModuleList([GPT2_Block(d_Embedding, dK, dV, heads, intermediate_size, dropout, device) for _ in range(n_blocks)])    

        # Calculate the number of parameters
        self.n_parameters=encoder.n_parameters + decoder.n_parameters + self.transformer_blocks[0].n_parameters

    def forward(self, x, edge_index):
        #Encoding
        x=self.encoder(x)
        
        #Transformer blocks
        for block in self.transformer_blocks:
            x=block(x, edge_index)

        #Decoding
        x=self.decoder(x)
        return x

    def load_from_original(self, pretrained_model):
        # Load the embedding layer
        self.encoder.load_from_original(pretrained_model.transformer)

        # Load the transformer blocks
        transformer_heads=pretrained_model.transformer.h
        for i in range(self.n_blocks):
            self.transformer_blocks[i].load_from_original(transformer_heads[i])

        # Load the language model head
        self.decoder.load_from_original(pretrained_model.lm_head)


class GPT2_Encoder(nn.Module):

    def __init__(self,d_Embedding=768, tokenizer=Tokenizer('gpt2'), max_position_encoding=1024, dropout=0.1, device='cpu'):
        super().__init__()

        # Save the parameters
        self.tokenizer=tokenizer
        self.d_Embedding=d_Embedding
        self.max_position_encoding=max_position_encoding
        self.device=device

        # Initialize the embedding layer
        self.embedding=nn.Embedding(tokenizer.vocab_size, d_Embedding, device=device)
        self.positional_encoding=nn.Embedding(max_position_encoding, d_Embedding, device=device)

        self.dropout=nn.Dropout(dropout)

        # Calculate the number of parameters
        self.n_parameters=tokenizer.vocab_size*d_Embedding + max_position_encoding*d_Embedding

    def forward(self, x):
        #tokenize if necessary
        if type(x)==str: x=self.tokenizer.encode(x)

        #Embedding
        indices=torch.arange(x.shape[0], device=self.device)
        x = self.embedding(x) + self.positional_encoding(indices)
        x = self.dropout(x)
        return x
    
    def load_from_original(self, trasformer):
        # Extract the submodules
        weight_token_embedding=trasformer.wte
        weight_positional_embedding=trasformer.wpe

        assert self.embedding.weight.shape==weight_token_embedding.weight.shape
        assert self.positional_encoding.weight.shape==weight_positional_embedding.weight.shape

        # Load the embedding layer
        self.embedding.weight=weight_token_embedding.weight
        self.positional_encoding.weight=weight_positional_embedding.weight

class GPT2_LM_Head(nn.Module):

    def __init__(self, d_Embedding=768, tokenizer=Tokenizer('gpt2'), device='cpu') -> None:
        super().__init__()

        self.d_Embedding=d_Embedding
        self.tokenizer=tokenizer
        self.device=device

        # Initialize the language model head
        self.layer_norm=nn.LayerNorm(d_Embedding, eps = 1e-5, elementwise_affine=True, device=device)
        self.language_model_head=nn.Linear(d_Embedding, tokenizer.vocab_size, bias=False, device=device)
        self.activation_head=nn.Softmax(dim=-1)

        # Calculate the number of parameters
        self.n_parameters=d_Embedding*tokenizer.vocab_size

    def forward(self, x):
        x=self.layer_norm(x)
        x=self.language_model_head(x)
        x=self.activation_head(x)
        return x
    
    def load_from_original(self, language_model_head):
        assert self.language_model_head.weight.shape==language_model_head.weight.shape
        assert self.language_model_head.bias.shape==language_model_head.bias.shape

        # Load the language model head
        self.language_model_head.weight=language_model_head.weight
        self.language_model_head.bias=language_model_head.bias

class GPT2_Block(nn.Module):

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
        self.attention_block.load_from_original(head.attn)

        # Load the MLP
        self.MLP.load_from_original(head.mlp)
        


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


    def load_from_original(self, attn):
        c_attn=attn.c_attn      
        w, b = c_attn.weight, c_attn.bias

        query_weight, key_weight, value_weight = torch.split(w, w.shape[-1]//3, dim=-1)
        query_bias,   key_bias,   value_bias   = torch.split(b, b.shape[-1]//3, dim=-1)

        assert self.key.weight.shape==key_weight.shape
        assert self.key.bias.shape==key_bias.shape
        self.key.weight=nn.Parameter(key_weight)
        self.key.bias=nn.Parameter(key_bias)

        assert self.query.weight.shape==query_weight.shape
        assert self.query.bias.shape==query_bias.shape
        self.query.weight=nn.Parameter(query_weight)
        self.query.bias=nn.Parameter(query_bias)

        assert self.value.weight.shape==value_weight.shape
        assert self.value.bias.shape==value_bias.shape
        self.value.weight=nn.Parameter(value_weight)
        self.value.bias=nn.Parameter(value_bias)

        c_proj=attn.c_proj
        
        assert self.feedforward.weight.shape==c_proj.weight.shape
        assert self.feedforward.bias.shape==c_proj.bias.shape
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


    def load_from_original(self,mlp):
        c_fc=mlp.c_fc
        c_proj=mlp.c_proj

        assert self.feedforward1.weight.shape==c_fc.weight.t().shape
        assert self.feedforward1.bias.shape  ==c_fc.bias.shape
        self.feedforward1.weight=nn.Parameter(c_fc.weight.t()) #it's the only way to make it work. Fuck who made this.
        self.feedforward1.bias  =nn.Parameter(c_fc.bias)

        assert self.feedforward2.weight.shape==c_proj.weight.t().shape
        assert self.feedforward2.bias.shape  ==c_proj.bias.shape
        self.feedforward2.weight=nn.Parameter(c_proj.weight.t())
        self.feedforward2.bias  =nn.Parameter(c_proj.bias)

import math
class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))