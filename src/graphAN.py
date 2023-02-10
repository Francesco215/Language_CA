
import torch
from torch import nn

from .encoder import Encoder
from .decoder import Decoder
from .transformerMP import TransformerBlock



class GraphAttentionNetwork(nn.Module):
    """
        The entire assembly for the graph attention network, it takes in a graph and returns a graph
    """
    def __init__(self, 
        encoder:Encoder,
        decoder:Decoder,
        transformer:TransformerBlock,
        transformer_layers:int=4,
        dK=1024, dV=1024, heads=8,
        ):
        
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder

        self.embedding_dim=encoder.embedding_dim

        transformer_blocks=[transformer(self.embedding_dim, dK, dV, heads) for _ in range(transformer_layers)]

        self.transformer = nn.Sequential(*transformer_blocks)

        self.n_parameters=encoder.n_parameters + decoder.n_parameters + transformer_blocks[0].n_parameters*transformer_layers

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
        
        for _ in range(iterations):
            x=self.transformer(x, edge_index)
        
        x=self.decoder(x)

        return x



