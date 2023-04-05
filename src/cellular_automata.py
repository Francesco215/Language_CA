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

        self.loss_function=loss_function
    
    def eval_loss(self, x, edge_index, noise, n_steps=10, starting_step=0, step_weight:callable=None):
        """Evaluates the loss of the graph attention network for many steps
        
            Args:
            x (torch.Tensor): Tokenized graph
                if x.dtype==torch.int64: x is a list of tokenized text
                else: x is a list of embeddings
            edge_index (torch.Tensor): Adjacency matrix of the graph
            target (torch.Tensor): Target graph
            n_steps (int, optional): Number of steps to take. Defaults to 10.
            starting_step (int, optional): Step to start the loss calculation. Defaults to 0.
            step_weight (callable, optional): Function that takes in the step number and returns a weight for that step. Defaults to None.

            Returns:  
            torch.Tensor: The loss scalar. dtype: torch.float
            torch.Tensor: The loss for each step. dtype: torch.float 
        """

        target = x.clone()
        if x.dtype==torch.long:
            x, clean_encoding, noise_encoding = self.encoder(x, noise)
        
        #initialize loss
        step_loss=torch.empty(n_steps, device=self.device)

        #do n steps 
        for i in range(n_steps):
            #make a forward pass
            x = super().forward(x, edge_index)
            #compute loss
            step_loss[i]=self.loss_function(x, target, clean_encoding, noise_encoding)
            #apply step weight if given
            if step_weight is not None:
                step_loss[i]*=step_weight(i+starting_step)

        return step_loss.mean(), step_loss
    


def simple_step_weight(step,starting_step=2):
    return step>=starting_step



class DiffusionLoss(Loss):
    def __init__(self, decoder: Decoder):
        loss_function=nn.CrossEntropyLoss()
        super().__init__(decoder, loss_function)
        self.embedding_loss=nn.MSELoss()


    def forward(self, x, target, clean_encoding, noise_encoding):
        return self.embedding_loss(x, clean_encoding) + self.loss(x-noise_encoding, target)