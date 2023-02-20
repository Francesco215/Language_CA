import torch
from torch import nn


class Decoder(nn.Module):

    def __init__(self,
        embedding_dim:int = 512,
        vocab_size: int = 28996, #defaults to the bert-base dictionary size
        ):

        super().__init__()


        self.layer=nn.Linear(embedding_dim,vocab_size)
        self.activation=nn.Softmax(dim=1)

        self.vocab_size=vocab_size
        self.embedding_dim=embedding_dim

        self.n_parameters=vocab_size*embedding_dim

    def forward(self,x):
        x=self.layer(x)
        x=self.activation(x)
        return x



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

