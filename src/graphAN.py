
import torch
from torch import nn

from .encoder import Encoder
from .decoder import Decoder
from .transformerMP import TransformerBlock
from .data_loader import Tokenizer



class GraphAttentionNetwork(nn.Module):
    """
        The entire assembly for the graph attention network, it takes in a graph and returns a graph
    """
    def __init__(self, 
        tokenizer:Tokenizer,
        encoder:Encoder,
        decoder:Decoder,
        transformer:TransformerBlock,
        transformer_layers:int=4,
        dK=1024, dV=1024, heads=8,
        ):
        
        super().__init__()
        
        self.tokenizer=tokenizer
        self.encoder = encoder
        self.decoder = decoder

        self.embedding_dim=encoder.embedding_dim

        self.transformer_blocks=[transformer(self.embedding_dim, dK, dV, heads) for _ in range(transformer_layers)]

        #self.transformer = nn.Sequential(*transformer_blocks)

        self.n_parameters=encoder.n_parameters + decoder.n_parameters + self.transformer_blocks[0].n_parameters*transformer_layers

    def forward(self, x, edge_index,iterations:int=1):
        """It takes in a graph and returns a graph

        Args:
            x (torch.Tensor): Tokenized graph
            edge_index (torch.Tensor): Adjacency matrix of the graph
            iterations (int, optional): Number of times the tranformers are applied to the graph.
                Defaults to 1.

        Returns:
            torch.Tensor: The predicted token for each node of the graph
        """

        x=self.encoder(x)
        
        for transformer in self.transformer_blocks:
            x=transformer(x, edge_index)
        
        return x

    def inference(self, x , edge_index, iterations:int=1):
        x=self.__call__(x,edge_index,iterations)
        x=self.decoder(x)
        x=x.argmax(dim=-1)
        x=self.tokenizer.decode(x)

        return x



