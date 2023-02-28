import torch
from torch import nn

from src.encoder import Encoder
import torch.nn.functional as F

class Decoder(nn.Module):
    #Adapted from https://stackoverflow.com/questions/57929299/how-to-share-weights-between-modules-in-pytorch

    def __init__(self,encoder: Encoder):
        super().__init__()
        self.encoder = encoder.embedding
        self.embedding_dim=encoder.embedding_dim
        self.vocab_size=encoder.vocab_size
        self.n_parameters=encoder.n_parameters

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.encoder.weight)

    @property
    def weight(self) -> torch.Tensor:
        return self.encoder.weight



class Loss(nn.Module):
    def __init__(self, decoder:Decoder):
        #TODO: add mask to the loss
        super().__init__()

        assert isinstance(decoder,Decoder), "decoder must be an instance of the Decoder class"
        self.decoder=decoder 

        self.loss=nn.CrossEntropyLoss() #TODO: check if it computes the loss in the correct channels

    def forward(self, x, y):
        """calculates the loss between the prediction x and the target y

        Args:
            x (torch.Tensor): prediction, has the shape (n_nodes, vocab_size). dtype: torch.float
            y (torch.Tensor): target, has the shape (n_nodes,). dtype: torch.long

        Returns:
            torch.Tensor: the loss scalar, has the shape (1,). dtype: torch.float
        """
        assert x.shape[0]==y.shape[0], "x and y must have the same number of nodes"
        assert x.shape[1]==self.decoder.embedding_dim, "x must have the same number of channels as the embedding dimension"
        assert y.max()<=self.decoder.vocab_size, "y must have the same number of channels as the vocabulary size"
        assert y.dtype==torch.long, "y must be of type torch.long"

        x=self.decoder(x)
        return self.loss(x,y)

