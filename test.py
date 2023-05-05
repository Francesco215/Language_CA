import torch
from torch import nn
from IPython.display import clear_output

from src.encoder import Encoder, GPT2Encoder
from src.decoder import Decoder, GPT2Decoder
from src.graph_initialization import random_unidirectional_graph_maker, linear_bidirectional_graph_maker
from src.graphAN import GraphAttentionNetwork, BlockGenerator
from src.data_loader import validation
from src.tokenizer import Tokenizer
from src.GPT2 import GPT2_Block, GPT2
from matplotlib import pyplot as plt
from src.utils import moving_average, grad_norm
from torch.nn.utils import clip_grad_norm_
import pickle
import numpy as np
from termcolor import colored
from torch.nn import functional as F
import einops

import torch

from src.cellular_automata import CellularAutomata
from src.tokenizer import CharTokenizer

dir_path = 'shakespeare_data/'
input_file_path = dir_path+'input.txt'


#create the tokenizer
tokenizer = CharTokenizer(input_file_path)
print('tokenizer vocab size:', tokenizer.vocab_size)

# load the data
with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")


# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = tokenizer(train_data)
val_ids = tokenizer(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
torch.save(train_ids, dir_path+'train.bin')
torch.save(val_ids,   dir_path+'val.bin')



from src.cellular_automata import DiffusionLoss
from src.decoder import Loss
from src.encoder import NoiseEncoder


device = 'cpu'
#device = 'mps'  if torch.backends.mps.is_available() else 'cpu'
device = 'cuda' if torch.cuda.is_available() else device

dK = 16
dV = 16
heads = 4
d_Embedding = 65
intermediate_size=2*d_Embedding

encoder = NoiseEncoder(d_Embedding, tokenizer, dropout=0, device=device, one_hot=True)
decoder = Decoder(encoder)
block_generator = BlockGenerator(GPT2_Block, d_Embedding, dK, dV, heads, intermediate_size,
                                 dropout=0.1, split_size=2**10, device=device, rotary_encoding=True)

model = CellularAutomata(tokenizer, encoder, block_generator, decoder, n_blocks=2)
model.logs={'loss':[],'step_loss':[], 'loss_components':[[],[],[]]}
model.tokens_seen=0

graph_maker = linear_bidirectional_graph_maker(64, device=device)

print(f'number of parameters:{model.n_parameters}')

lr = 2e-2
gamma = 0.99

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

step_weight=None


loss_function=DiffusionLoss(decoder,1e-2,1e-1)



from numpy.random import randint
def sample_shakespeare(data, lenght, starting_index=None):
    lenght=int(lenght)
    
    if starting_index is None:
        starting_index = randint(0, len(data)-lenght)

    if starting_index+lenght>=len(data):
        return data[starting_index:]    
    
    return data[starting_index:starting_index+lenght]


def sample_minibatched_shakespeare(train_ids, context_size, batch_size):
    target = []
    prediction = []
    clean_encoding = []
    noise_encoding = []

    for _ in range(batch_size):
        targ=sample_shakespeare(train_ids, context_size)
        target.append(targ)
        noise=torch.rand(())

        pred, clean, nois = encoder(targ, noise)

        prediction.append(pred)
        clean_encoding.append(clean)
        noise_encoding.append(nois.repeat(pred.shape[0], 1))
        
    target = torch.cat(target, dim=0)
    prediction = torch.cat(prediction, dim=0)
    clean_encoding = torch.cat(clean_encoding, dim=0)
    noise_encoding = torch.cat(noise_encoding, dim=0)

    return prediction, clean_encoding, noise_encoding, target


batch_size=10

n_epochs = int(2000)
model.train()
context_size=100
model.train()
n_steps=10

edge_index = graph_maker(context_size)
edge_slide = torch.repeat_interleave(torch.arange(batch_size)*context_size, edge_index.shape[-1]).repeat(2,1)
edge_index=edge_index.repeat(1, batch_size) + edge_slide



prediction, clean_encoding, noise_encoding, target = sample_minibatched_shakespeare(train_ids, context_size, batch_size)


#do n steps 
step_loss, loss_components=loss_function(prediction, target, clean_encoding, noise_encoding)