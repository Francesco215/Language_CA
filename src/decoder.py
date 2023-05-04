import torch
from torch import nn
from src.tokenizer import Tokenizer

from src.encoder import Encoder, GPT2Encoder, NoiseEncoder
import torch.nn.functional as F

class Decoder(nn.Module):
    #Adapted from https://stackoverflow.com/questions/57929299/how-to-share-weights-between-modules-in-pytorch

    def __init__(self,encoder: Encoder):
        super().__init__()

        assert isinstance(encoder, Encoder), "encoder must be of type Encoder"

        self.encoder = encoder
        self.d_Embedding=encoder.d_Embedding
        self.vocab_size=encoder.vocab_size
        self.n_parameters=encoder.n_parameters
        self.device=encoder.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)

    @property
    def weight(self) -> torch.Tensor:
        return self.encoder.embedding.weight



class Loss(nn.Module):
    def __init__(self, decoder:Decoder, loss_function=nn.CrossEntropyLoss()):
        """Initializes the loss class

        Args:
            decoder (Decoder): The decoder function
            loss_function (nn.Module, optional): The actual loss funciton(x,y).
                Defaults to nn.CrossEntropyLoss().
        """
        super().__init__()

        self.decoder=decoder
        self.encoder=decoder.encoder

        self.loss=loss_function #TODO: check if it computes the loss in the correct channels

    def forward(self, embeddings, targets):
        """calculates the loss between the predicted embeddigns and the targets

        Args:
            embeddings (torch.Tensor): predicted embeddings, has the shape (n_nodes, vocab_size). dtype: torch.float
            targets (torch.Tensor): target, has the shape (n_nodes,). dtype: torch.long

        Returns:
            torch.Tensor: the loss scalar. dtype: torch.float
        """
        assert embeddings.shape[0]==targets.shape[0], "x and y must have the same number of nodes"
        assert embeddings.shape[1]==self.decoder.d_Embedding, "x must have the same number of channels as the embedding dimension"
        assert targets.max()<=self.decoder.vocab_size, "y must have the same number of channels as the vocabulary size"
        assert targets.dtype==torch.long, "y must be of type torch.long"

        embeddings=self.decoder(embeddings)
        return self.loss(embeddings, targets)


class DinstinctionLoss(nn.Module):
    """
        Not used, for now its not useful
    
        This is a loss function that makes sure that the encoder encodes the inputs
        in such a way that are distinguishable for the decoder.

    """

    def __init__(self,decoder: Decoder):

        super().__init__()

        self.decoder=decoder
        self.encoder=decoder.encoder

        self.loss=nn.CrossEntropyLoss()


    def forward(self, targets, clean_encoding=None):
        assert targets.dtype==torch.long, "the targets must be of type torch.long"

        if clean_encoding==None:
            clean_encoding=self.encoder(targets)
            if isinstance(self.encoder, NoiseEncoder):
                clean_encoding=clean_encoding[1]-clean_encoding[2]
        
        out=self.decoder(clean_encoding)
        return self.loss(out, targets)


class GPT2Decoder(Decoder):
    def __init__(self, encoder:GPT2Encoder):
        super().__init__(encoder)

        # Initialize the language model head
        self.layer_norm=nn.LayerNorm(self.d_Embedding, eps = 1e-5, elementwise_affine=True, device=self.device)
        # Calculate the number of parameters
        self.n_parameters+=self.layer_norm.weight.numel()+self.layer_norm.bias.numel()

    def forward(self, x):
        x = self.layer_norm(x)
        x = super().forward(x)

        return x
    
    def load_from_original(self, ln_f):
        self.layer_norm=ln_f