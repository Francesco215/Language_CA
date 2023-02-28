
import torch
from torch import nn

from .encoder import Encoder
from .decoder import Decoder
from .transformerMP import AttentionBlock
from .data_loader import Tokenizer


class GraphAttentionNetwork(nn.Module):
    """
        The entire assembly for the graph attention network, it takes in a graph and returns a graph
    """
    def __init__(self, 
        tokenizer:Tokenizer,
        encoder:Encoder,
        transformer:AttentionBlock,
        transformer_layers:int=4,
        dK=1024, dV=1024, heads=8,
        ):
        
        super().__init__()

        assert     isinstance(tokenizer,  Tokenizer), "tokenizer must be an instance of the Tokenizer class"
        assert     isinstance(encoder,    Encoder), "encoder must be an instance of the Encoder class"
        assert not isinstance(transformer,AttentionBlock), "transformer cannot be an instance of the TransformerBlock class"

        self.tokenizer=tokenizer
        
        self.encoder = encoder
        self.embedding_dim=encoder.embedding_dim
        
        self.decoder = Decoder(self.encoder)

        self.transformer_blocks=nn.ModuleList([transformer(self.embedding_dim, dK, dV, heads) for _ in range(transformer_layers)])

        #self.transformer = nn.Sequential(*transformer_blocks)

        self.n_parameters=encoder.n_parameters + self.decoder.n_parameters + self.transformer_blocks[0].n_parameters*transformer_layers

    def forward(self, x, edge_index,iterations:int=1):
        """It takes in a graph and returns a graph

        Args:
            x (torch.Tensor): Tokenized graph
                if x.dtype==torch.int64: x is a list of tokenized text
                else: x is a list of embeddings
            edge_index (torch.Tensor): Adjacency matrix of the graph
            iterations (int, optional): Number of times the tranformers are applied to the graph.
                Defaults to 1.

        Returns:
            torch.Tensor: The predicted token for each node of the graph
        """

        if (x.dtype==torch.long): x=self.encoder(x)

        for _ in range(iterations):
            for transformer in self.transformer_blocks:
                x=transformer(x, edge_index)
        
        return x

    @torch.no_grad()
    def inference(self, x , edge_index, iterations:int=1):
        x=self.__call__(x,edge_index,iterations)
        x=self.decoder(x)
        x=x.argmax(dim=-1)
        x=self.tokenizer.decode(x)

        return x



