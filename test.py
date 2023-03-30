from src.decoder import Loss
import os
import requests
import torch
from torch import nn
from IPython.display import clear_output

from src.encoder import Encoder, GPT2Encoder
from src.decoder import Decoder, GPT2Decoder
from src.graph_initialization import random_unidirectional_graph_maker, linear_unidirectional_graph_maker
from src.graphAN import GraphAttentionNetwork, BlockGenerator
from src.data_loader import validation
from src.tokenizer import Tokenizer
from src.GPT2 import GPT2_Block, GPT2
from matplotlib import pyplot as plt
from src.utils import moving_average, grad_norm
from torch.nn.utils import clip_grad_norm_
import pickle
import numpy as np


from src.tokenizer import CharTokenizer

__file__='shakespeare_data/'

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')


#create the tokenizer
tokenizer = CharTokenizer(input_file_path)
#print('tokenizer vocab size:', tokenizer.vocab_size)


# load the data
with open(input_file_path, 'r') as f:
    data = f.read()
#print(f"length of dataset in characters: {len(data):,}")


# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = tokenizer(train_data)
val_ids = tokenizer(val_data)
#print(f"train has {len(train_ids):,} tokens")
#print(f"val has {len(val_ids):,} tokens")

# export to bin files
torch.save(train_ids, os.path.join(os.path.dirname(__file__), 'train.bin'))
torch.save(val_ids,   os.path.join(os.path.dirname(__file__), 'val.bin'))

device = 'cpu'
#device = 'mps'  if torch.backends.mps.is_available() else 'cpu'
device = 'cuda' if torch.cuda.is_available() else device

dK = 16
dV = 16
heads = 6
d_Embedding = dK*heads
intermediate_size=intermediate_size=2*d_Embedding


encoder = Encoder(d_Embedding, tokenizer, dropout=0, device=device)
decoder = Decoder(encoder)
block_generator = BlockGenerator(GPT2_Block, d_Embedding, dK, dV, heads, intermediate_size,
                                 dropout=0.1, split_size=2**10, device=device, rotary_encoding=True)

model = GraphAttentionNetwork(tokenizer, encoder, block_generator, decoder)
model.losses = []
model.validation_losses = []

model.load('shakespeare_data/pretrained_CE=1.3.pth')


graph_maker = random_unidirectional_graph_maker(50, 50, device=device)

loss_function = Loss(decoder)
lr = 1e-9
gamma = 0.99

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)


from numpy.random import randint
def sample_shakespeare(data, lenght, starting_index=None):
    lenght=int(lenght)
    
    if starting_index is None:
        starting_index = randint(0, len(data)-lenght)

    if starting_index+lenght>=len(data):
        return data[starting_index:]    
    
    return data[starting_index:starting_index+lenght]


model.eval()

#data loading
text=sample_shakespeare(val_data, 100) #sample 100 characters data from the validation set
embeddings=tokenizer(text)# and tokenize it

target=embeddings[1:] # the target is the same as the input but shifted by one
embeddings=embeddings[:-1]# the input is the same as the target but shifted by one
edge_index=graph_maker(embeddings.shape[0]) #this in not really important for now what it means


#inference and loss
logits=model.final_embedding(embeddings,edge_index) #calculate the prediction of the final embedding
ce_loss=nn.CrossEntropyLoss()
print('validation cross_entropy:',ce_loss(logits,target).item()) #returns about ~1.5


#text generation
print('\ninput text:\n',text)
for _ in range(40): #lets generate 40 characters
    edge_index=graph_maker(embeddings.shape[0]) #this in not really important for now what it means
    logits=model.final_embedding(embeddings,edge_index) #calculate the prediction of the final embedding
    logits=logits[-1] #take the last prediction
    last_token=torch.argmax(logits) #take the token with the highest probability

    embeddings=torch.cat((embeddings,last_token.unsqueeze(0))) #add the token to the input


print('\n\ngenerated text:') 
print(tokenizer.decode(embeddings[-40:])) #print the generated text




logits=model.final_embedding(embeddings,edge_index)

loss=F.crossentropy(logits,target)


out=model(embeddings,edge_index)

loss_function(out,target)