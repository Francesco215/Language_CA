from torch import nn
import torch, transformers
from src import Tokenizer


tokenizer = Tokenizer('gpt2')
pretraied = transformers.GPT2LMHeadModel.from_pretrained('gpt2')


sequence_length = 17
batch_size = 1
n_heads = 12
d_Embedding = 64*n_heads

ln_1 = nn.LayerNorm(d_Embedding, eps=1e-5)
ln_2 = nn.LayerNorm(d_Embedding, eps=1e-5)


def forward(hidden_states):

    residual = hidden_states  # residual

    hidden_states = ln_1(hidden_states)  # normalization

    ln1=hidden_states.clone()

    attn_outputs = pretraied.transformer.h[0].attn(hidden_states)
    attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
    outputs = attn_output.clone()
    # residual connection
    hidden_states = attn_output + residual  # attention

    residual = hidden_states  # reisdual again

    hidden_states = ln_2(hidden_states)  # normalization
    ln2=hidden_states.clone()

    feed_forward_hidden_states = pretraied.transformer.h[0].mlp(
        hidden_states)  # mlp
    # residual connection
    hidden_states = residual + feed_forward_hidden_states

    outputs = (hidden_states,attn_output,ln1,ln2) 

    #in my model it only return hidden states
    # hidden_states, present, (attentions, cross_attentions)
    return outputs


x = torch.randn((1, sequence_length, d_Embedding))
y=x.clone()
out1 = forward(x)
block=pretraied.transformer.h[0]
block.final=out1[0]
block.attn_output=out1[1]
block.ln1=out1[2]
block.ln2=out1[3]
block.inp=y
out2 = block(y)


print('asd')