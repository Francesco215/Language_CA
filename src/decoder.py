import torch
from torch import nn
from src.data_loader import Tokenizer

from src.encoder import Encoder
import torch.nn.functional as F

class Decoder(nn.Module):
    #Adapted from https://stackoverflow.com/questions/57929299/how-to-share-weights-between-modules-in-pytorch

    def __init__(self,encoder: Encoder):
        super().__init__()
        self.encoder = encoder.embedding
        self.d_Embedding=encoder.d_Embedding
        self.vocab_size=encoder.vocab_size
        self.n_parameters=encoder.n_parameters
        self.device=encoder.device

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
        assert x.shape[1]==self.decoder.d_Embedding, "x must have the same number of channels as the embedding dimension"
        assert y.max()<=self.decoder.vocab_size, "y must have the same number of channels as the vocabulary size"
        assert y.dtype==torch.long, "y must be of type torch.long"

        x=self.decoder(x)
        return self.loss(x,y)



class GPT2Decoder(nn.Module):
    def __init__(self, d_Embedding=768, tokenizer=Tokenizer('gpt2'), device='cpu') -> None:
        super().__init__()

        self.d_Embedding=d_Embedding
        self.tokenizer=tokenizer
        self.device=device

        # Initialize the language model head
        self.layer_norm=nn.LayerNorm(d_Embedding, eps = 1e-5, elementwise_affine=True, device=device)
        self.language_model_head=nn.Linear(d_Embedding, tokenizer.vocab_size, bias=False, device=device)

        # Calculate the number of parameters
        self.n_parameters=d_Embedding*tokenizer.vocab_size

    def forward(self, x):
        x=self.layer_norm(x)
        x=self.language_model_head(x)

        return x
    
    def load_from_original(self, ln_f, language_model_head):
        self.layer_norm=ln_f

        assert self.language_model_head.weight.shape==language_model_head.weight.shape

        # Load the language model head
        self.language_model_head.weight=language_model_head.weight