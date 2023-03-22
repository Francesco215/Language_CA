import torch
from torch import nn
import einops

from src.tokenizer import Tokenizer

class Encoder(nn.Module):
    """Layer to turn tokens into word embeddings"""

    def __init__(self,
                 d_Embedding: int = 512,
                 tokenizer=Tokenizer('gpt2'),  # defaults to the bert-base dictionary size,
                 dropout: float = 0.1,
                 device: str = 'cpu'
                 ):
        """
        This funciton takes the tokens and turns them into word embeddings.
        if the input is a string, it will be tokenized using the tokenizer.
         

        Args:
            d_embedding (int):The size of the embedding vector.
            tokenizer (Tokenizer, optional): The tokenizer to use. Defaults to Tokenizer('gpt2').
            dropout (float, optional): The dropout rate. Defaults to 0.1.
            device (str, optional): The device to use. Defaults to 'cpu'.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.d_Embedding = d_Embedding
        self.device=device

        #the embedding layer turns each token into a vector of size d_Embedding
        self.embedding = nn.Embedding(self.vocab_size, d_Embedding,device=device)
        self.dropout = nn.Dropout(dropout)

        #the number of parameters is the number of tokens times the embedding dimention
        self.n_parameters = self.vocab_size * d_Embedding

    def forward(self, x):
        #tokenize if necessary
        if type(x) == str:
            x = self.tokenizer(x)

        x = self.embedding(x)
        x = self.dropout(x)
        return x




class GPT2Encoder(Encoder):

    def __init__(self, d_Embedding=768, tokenizer=Tokenizer('gpt2'), max_position_encoding=1024, dropout=0.0, device='cpu'):
        super().__init__(d_Embedding,tokenizer,dropout,device)

        # Save the parameters
        self.max_position_encoding = max_position_encoding

        # Add positional encoding
        self.positional_encoding = nn.Embedding(max_position_encoding, d_Embedding, device=device)

        # Calculate the number of parameters
        self.n_parameters = tokenizer.vocab_size * d_Embedding + max_position_encoding * d_Embedding

    def forward(self, x):
        #tokenize if necessary
        if type(x) == str:
            x = self.tokenizer(x)

        assert x.shape[0]<self.max_position_encoding, f"The sequence is too long for the positional encoding, got {x.shape[0]} but the maximum is {self.max_position_encoding}"

        #Embedding
        indices = torch.arange(x.shape[0], device=self.device)
        x = self.embedding(x) + self.positional_encoding(indices)
        x = self.dropout(x)
        return x

    def load_from_original(self, trasformer):
        # Extract the submodules
        weight_token_embedding = trasformer.wte
        weight_positional_embedding = trasformer.wpe

        assert self.embedding.weight.shape == weight_token_embedding.weight.shape
        assert self.positional_encoding.weight.shape == weight_positional_embedding.weight.shape

        # Load the embedding layer
        self.embedding.weight = weight_token_embedding.weight
        self.positional_encoding.weight = weight_positional_embedding.weight






def rotary_encoding(x, base=1e-5, thetas=None):
    """Applies a rotary embedding to a tensor.

    Args:
        x (torch.Tensor): Tensor to apply the rotary embedding to.
        base (float, optional): Base of the logarithm. Defaults to 1e-5.
        thetas (torch.Tensor, optional): Tensor containing the thetas.
            It can be used in case you want to apply learned positional encoding.
            Defaults to None.

    Returns:
        torch.Tensor: Tensor with the rotary embedding applied.
    """

    #pad with zeros if odd, otherwise we cant pair up consecutive elements
    odd = False
    if x.shape[0] % 2 != 0:
        zeros = torch.zeros((1, *x.shape[1:]), device=x.device)
        x = torch.cat([x, zeros], dim=0)
        odd = True

    #pair up consecutive elements
    x1 = einops.rearrange(x, '(n1 n2) ... -> n1 n2 ...', n2=2)

    #pair up elements and swap them
    x2 = x1[:, torch.tensor([1, 0])]
    x2[:, 0] = -x2[:, 0]

    #create phases
    sin, cos = make_sin_cos(x1.shape,base,thetas,device=x.device)

    #apply rotation
    x1 = einops.einsum(x1, cos, 'a ... c, a c -> a ... c')
    x2 = einops.einsum(x2, sin, 'a ... c, a c -> a ... c')
    x = x1+x2
    x = einops.rearrange(x, 'n1 n2 ...->(n1 n2) ...')

    if odd:
        return x[:-1]  # remove padding if odd
    return x


def make_sin_cos(shape, base=1e-5, thetas=None, device='cpu'):    
    if thetas is None:
        thetas = torch.logspace(0, 1, shape[-1], base, device=device)
    indices = torch.arange(0, shape[0], device=device)
    phases = einops.einsum(indices, thetas, 'a, c -> a c')

    #rotate
    sin = torch.sin(phases)
    cos = torch.cos(phases)

    #zetas = torch.linspace(gamma,2*x.shape[-1]+gamma,x.shape[-1],device=x.device)/(1+gamma)

    return sin, cos