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

from src.cellular_automata import DiffusionLoss
from src.decoder import Loss
from src.encoder import NoiseEncoder
from src.training_pool import SamplePool, TextGenerator


dir_path = 'shakespeare_data/'
input_file_path = dir_path+'input.txt'


#create the tokenizer
tokenizer = CharTokenizer(input_file_path)
print('tokenizer vocab size:', tokenizer.vocab_size)

# load the data

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

sequence_lenght=100
text_generator=TextGenerator(input_file_path, lenght=sequence_lenght, encoder=encoder)
pool=SamplePool(pool_size=30, generator=text_generator, encoder=encoder, indexes_max_loss_size=5)


#@title {vertical-output: true}
#@markdown # Training
#@markdown the loss function is cross entropy ❌🎲

edge_index = graph_maker(sequence_lenght)


n_epochs = int(2000)
model.train()
n_steps=10
for i in range(n_epochs):
    idx=np.random.randint(pool.size)
    print(idx)
    target_tokens, prediction, clean_embeddings, noise_encoding, noise_level = pool[idx]

    step_loss=torch.empty(n_steps+1, device=device)
    loss_components=torch.empty([n_steps+1,3], device=device)

    #do n steps 
    step_loss[0], loss_components[0]=loss_function(prediction, target_tokens, clean_embeddings, noise_encoding)
    for j in range(1,n_steps+1):
        #make a forward pass
        prediction = model(prediction, edge_index)
        
        #compute loss
        step_loss[j], loss_components[j] = loss_function(prediction, target_tokens, clean_embeddings, noise_encoding)
        #apply step weight if given
        if step_weight is not None:
            step_loss[j]*=step_weight(j)

    #compute the total loss

    loss = step_loss.mean()

    model.logs['loss'].append(loss.item())
    model.logs['step_loss'].append(step_loss.detach().cpu().numpy())
    for log,l in zip(model.logs['loss_components'],loss_components.mean(dim=0)):
        log.append(l.item())

    
    loss.backward()
    clip_grad_norm_(model.parameters(), 4*loss.item())


    optimizer.step()
    optimizer.zero_grad()  # reinitialize the gradient to zero
    model.tokens_seen+=sequence_lenght


    logging_interval=30
    if i%logging_interval==logging_interval-1:
        clear_output(wait=True)
        
        m_av = moving_average(model.logs['loss'], logging_interval-1)
        
        plt.plot(model.logs['loss'], label='loss', color='grey', alpha=0.5, linewidth=0.5)
        plt.plot(m_av, label='moving average', color='black')

        
        plt.title("training loss")
        plt.legend()
        plt.ylabel('loss')
        plt.xlabel('iteration')
        plt.yscale('log')
        plt.xscale('log')
        plt.show()

        prediction, clean_embeddings, noise_encoding, target_tokens = sample_minibatched_shakespeare(train_ids, sequence_lenght, batch_size)

        

        plt.plot(model.logs['loss'], label='loss', color='grey', alpha=0.5, linewidth=0.5)
        plt.plot(m_av, label='moving average', color='black')

        labels=["mse","reconstruction","dinstinction"]
        colors=[["red","orange"], ["blue", "cyan"], ["green", "lime"]]
        for log, label, color in zip(model.logs['loss_components'], labels, colors):
            m_av = moving_average(log, logging_interval-1)
            plt.plot(m_av, label=f'moving {label}', color=color[0])
            plt.plot(log, label=label, color=color[1], alpha=0.5, linewidth=0.5)



        plt.title("validation")
        plt.legend()
        plt.ylabel('loss')
        plt.xlabel('iteration')
        plt.yscale('log')
        plt.xscale('log')
        plt.ylim(bottom=5e-2)
        plt.show()
        