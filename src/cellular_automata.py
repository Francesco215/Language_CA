from src.tokenizer import Tokenizer
from src.encoder import Encoder, NoiseEncoder
from src.decoder import Decoder, Loss


import torch
import torch.nn as nn

from src.graphAN import BlockGenerator, GraphAttentionNetwork


class CellularAutomata(GraphAttentionNetwork):

    def __init__(self, 
        tokenizer:Tokenizer,
        encoder:NoiseEncoder,
        block_generator:BlockGenerator,
        decoder:Decoder=None,
        n_blocks:int=4,
        loss_function=None,
        ):
        
        assert isinstance(encoder,NoiseEncoder), "encoder must be of type NoiseEncoder"
        super().__init__(tokenizer, encoder, block_generator, decoder, n_blocks)

        if loss_function==None: self.loss_function=DiffusionLoss(self.decoder)
        else: self.loss_function=loss_function

        

    def denoise(self, x, edge_index, noise =torch.tensor(1.), n_steps=10, starting_step=0, step_weight:callable=None):
        """Denoises a graph
        
            Args:
            x (torch.Tensor): Tokenized graph
                if x.dtype==torch.int64: x is a list of tokenized text
                else: x is a list of embeddings
            edge_index (torch.Tensor): Adjacency matrix of the graph
            noise (float | torch.Tensor): Noise level for the encoder
            n_steps (int): Number of steps to do
            starting_step (int): Starting step for the step weight
            step_weight (callable): Function that takes the step number and returns a weight

            Returns:  
            torch.Tensor: The denoised graph
        """

        if x.dtype==torch.long:
            x, clean_encoding, noise_encoding = self.encoder(x, noise)
        
        #do n steps 
        for i in range(n_steps):
            #make a forward pass
            x = self.forward(x, edge_index)
            #apply step weight if given
            if step_weight is not None:
                x = x*step_weight(i+starting_step) + clean_encoding*(1-step_weight(i+starting_step))

        return x

def simple_step_weight(step,starting_step=2):
    return step>=starting_step



class DiffusionLoss(Loss):
    def __init__(self, decoder: Decoder):
        loss_function=nn.CrossEntropyLoss()
        super().__init__(decoder, loss_function)
        self.embedding_loss=nn.MSELoss()


    def forward(self, x, target, clean_encoding, noise_encoding):
        return self.embedding_loss(x, clean_encoding) + self.loss(x-noise_encoding, target)