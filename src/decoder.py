import torch
from torch import nn


class Decoder(nn.Module):

    def __init__(self,
        hidden_dim: int = 2048,
        embedding_dim:int = 512,
        dictionary_size: int = 28996, #defaults to the bert-base dictionary size
        ):

        self.layers=nn.Sequential(
            nn.Linear(embedding_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,dictionary_size),
            nn.Softmax(dim=-1)
        )

    def forward(self,x):
        return self.layers(x)