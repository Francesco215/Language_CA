
import torch
from torch import nn

from .encoder import Encoder
from .decoder import Decoder
from .data_loader import Tokenizer

class BlockGenerator:
    def __init__(self, block, *args, **kwargs):
        """Generates a block of the graph attention network

        Args:
            block : class initializer
        """
        self.block = block

        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        out = self.block(*self.args, **self.kwargs)
        assert isinstance(out,nn.Module)
        return out
    


class GraphAttentionNetwork(nn.Module):
    """
        The entire assembly for the graph attention network, it takes in a graph and returns a graph
    """
    def __init__(self, 
        tokenizer:Tokenizer,
        encoder:Encoder,
        block_generator:BlockGenerator,
        decoder:Decoder=None,
        n_blocks:int=12,
        ):
        
        super().__init__()

        assert isinstance(tokenizer,   Tokenizer), "tokenizer must be an instance of the Tokenizer class"
        assert isinstance(block_generator, BlockGenerator), "transformer cannot be an instance of the TransformerBlock class"

        self.tokenizer=tokenizer
        
        self.encoder = encoder
        self.d_Embedding=encoder.d_Embedding
        self.n_blocks=n_blocks
        self.device=encoder.device

        if decoder is None:        
            self.decoder = Decoder(self.encoder)
        else:
            self.decoder = decoder

        self.transformer_blocks=nn.ModuleList([block_generator() for _ in range(n_blocks)])

        self.n_parameters=encoder.n_parameters + self.decoder.n_parameters + self.transformer_blocks[0].n_parameters*n_blocks

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
    
    def calculate_final_embedding(self, x, edge_index, iterations:int=1):
        x=self.__call__(x,edge_index,iterations)
        x=self.decoder(x)

        return x

    @torch.no_grad()
    def inference(self, x , edge_index, iterations:int=1):
        x=self.calculate_final_embedding(x,edge_index,iterations)
        x=x.argmax(dim=-1)
        x=self.tokenizer.decode(x)

        return x


