import torch
from torch import nn

from src import  GraphAttentionNetwork

#GPT2_Block(d_Embedding, dK, dV, heads, intermediate_size, dropout, device)
class GPT2(GraphAttentionNetwork):
    
    def load_from_original(self, pretrained_model):
        # Load the embedding layer
        pretrained_model.to(self.device)
        self.encoder.load_from_original(pretrained_model.transformer)

        # Take the transformer blocks
        transformer_heads=pretrained_model.transformer.h
        assert len(transformer_heads)==self.n_blocks, f"n_blocks must be equal to the number of trasnformer heads"
        # Load the transformer blocks
        for i in range(self.n_blocks):
            self.transformer_blocks[i].load_from_original(transformer_heads[i])

        # Load the language model head
        self.decoder.load_from_original(pretrained_model.transformer.ln_f, pretrained_model.lm_head)





class GPT2_Block(nn.Module):

    def __init__(self, d_Embedding=768, dK=64, dV=64, heads=12,intermediate_size=3072, dropout=0.0, device='cpu', split_size=2**15):
        super().__init__()

        # Save the parameters
        self.d_Embedding=d_Embedding
        self.dK=dK
        self.dV=dV
        self.heads=heads
        self.intermediate_size=intermediate_size
        self.dropout=dropout
        self.device=device
        self.split_size=split_size
        
        # Initialize the transformer block and the MLP
        self.attention_block=AttentionBlockGPT2(d_Embedding, dK, dV, heads, dropout, device, split_size)
        self.MLP=GPT2MLP(d_Embedding, intermediate_size, dropout, device)

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(self.d_Embedding, eps = 1e-5, elementwise_affine=True, device=device)
        self.layer_norm2 = nn.LayerNorm(self.d_Embedding, eps = 1e-5, elementwise_affine=True, device=device)

        # Calculate the number of parameters
        self.n_parameters=self.attention_block.n_parameters + self.MLP.n_parameters

    def forward(self, x, edge_index):
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
        # Copy LayerNorm layers
        self.layer_norm1=head.ln_1
        self.layer_norm2=head.ln_2

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
    def load_from_original(self, attn):

        Conv1D_to_Linear(attn.c_attn,self.make_QKV)        
        Conv1D_to_Linear(attn.c_proj, self.feedforward)    




class GPT2MLP(nn.Module):
    def __init__(self, d_Embedding=768, intermediate_size=3072, dropout=0.0, device='cpu'):
        super().__init__()
        
        self.feedforward1=nn.Linear(d_Embedding, intermediate_size, device=device)
        self.feedforward2=nn.Linear(intermediate_size, d_Embedding, device=device)

        self.dropout=nn.Dropout(dropout)
        self.activation=NewGELUActivation()

        self.n_parameters=2*d_Embedding*intermediate_size

    def forward(self,x):
        x=self.feedforward1(x)
        x=self.activation(x)
        x=self.feedforward2(x)
        x=self.dropout(x)

        return x


    def load_from_original(self,mlp):
        c_fc=mlp.c_fc
        c_proj=mlp.c_proj

        Conv1D_to_Linear(c_fc,self.feedforward1)
        Conv1D_to_Linear(c_proj,self.feedforward2)






from transformers.modeling_utils import Conv1D
def Conv1D_to_Linear(conv:Conv1D,lin:nn.Linear):
    """This is to convert the Conv1D layers that appear in the hugging face source code of GPT2
    to nn.Linear layers present in my implementation

    Args:
        conv (Conv1D): Conv1D layers
        lin (nn.Linear): Linear layer
    """

    assert isinstance(conv,Conv1D), 'conv must be of type transformers.modeling_utils.Conv1D'
    assert isinstance(lin,nn.Linear), 'lin must be of type torch.nn.Linear'

    weight = conv.weight.t()
    bias   = conv.bias

    assert lin.weight.shape==weight.shape, f'The two weight shapes are incompatible.\n\tConvolutional layer shape = {weight.shape}\n\tLinear layer shape = {lin.weight.shape}'
    assert lin.bias.shape  ==bias.shape,   'the two bias shapes are incompatible'

    lin.weight = nn.Parameter(weight)
    lin.bias   = nn.Parameter(bias)

import math
class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))