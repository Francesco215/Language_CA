from src.tokenizer import Tokenizer
from src.encoder import Encoder, NoiseEncoder
from src.decoder import Decoder, DinstinctionLoss, Loss


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
    def __init__(self, decoder: Decoder, decoder_loss_weight=1, dinstinction_loss_weight=1):
        """Initializes the loss class for the diffusion model

        Args:
            decoder (Decoder): The decoder function
            decoder_loss_weight (float, optional): The weight of the decoder loss.
                It determines how much the decoder loss contributes to the total loss.
                Defaults to 1.
            dinstinction_loss_weight (float, optional): The weight of the dinstinction loss.
                the dinstinction loss makes sure that the encoder encodes the inputs in such
                a way that are distinguishable for the decoder. 
                Defaults to 1.
        """
        loss_function=nn.CrossEntropyLoss()
        super().__init__(decoder, loss_function)
        assert isinstance(self.encoder, NoiseEncoder), f"The encoder must be a of type NoiseEncoder, instead got {type(self.encoder)}"

        self.embedding_loss=nn.MSELoss()

        self.weights=torch.tensor([1, decoder_loss_weight, dinstinction_loss_weight])
        self.weights=self.weights/self.weights.sum()

    def forward(self, encoding_prediction, target, clean_encoding, noise_encoding):
        """Evaluates the simplified loss function for the diffusion model with a given decoder
        
        Args:
            encoding_prediction (torch.Tensor): the prediction of the encoding (dtype=torch.float)
            target (torch.Tensor): the target of the decoder (dtype=torch.long)
            clean_encoding (torch.Tensor): the encoding without noise applied (dtype=torch.float)
            noise_encoding (torch.Tensor): the encoding of the noise itself (dtype=torch.float)

        Returns:
            torch.Tensor: The loss
        """
        losses=torch.empty([3])

        #MSE loss over the embedding space
        losses[0] = self.embedding_loss(encoding_prediction, clean_encoding)
        
        #The loss relative to the decoding
        logits=self.decoder(encoding_prediction-noise_encoding)
        losses[1] = self.loss(logits, target)
        
        #This is a loss function that makes sure that the encoder encodes the inputs
        #in such a way that are distinguishable for the decoder.
        clean_logits=self.decoder(clean_encoding-noise_encoding)
        losses[2] = self.loss(clean_logits,target)

        total_loss=torch.dot(losses,self.weights)

        return total_loss, losses