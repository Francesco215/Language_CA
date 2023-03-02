import torch
from torch import nn

from src.data_loader import Tokenizer

class Encoder(nn.Module):
    """Layer to turn tokens into word embeddings, it also supports positional embeddings"""

    def __init__(self,
                 d_Embedding: int = 512,
                 base_freq: float = 1e-5,
                 vocab_size: int = 28996,  # defaults to the bert-base dictionary size,
                 dropout: float = 0.1,
                 device: str = 'cpu'
                 ):
        """We first turn tokens into embedding via the self.emb function which turns each token,
        which is a scalar, into n_embedding dimentional vector.
        The input vector has size where L is the sequence lenght.
         

        Args:
            d_embedding (int):The size of the embedding vector.
            base_freq (float, optional): The base frequency of sinusoidal
                functions for the positional encoding. (default: 1e-4)
            vocab_size (int, optional): The size of the dictionary of the tokenizer.
                default: 28996 which is equal to the bert-base dictionary size
            dropout (float, optional): The dropout rate. Defaults to 0.1.
                TODO: chec if it should be applied to the positional encoding.
        """
        super().__init__()

        #the embedding layer turns each token into a vector of size d_Embedding
        self.embedding = nn.Embedding(vocab_size, d_Embedding,device=device)
        self.dropout = nn.Dropout(dropout)

        self.vocab_size = vocab_size
        self.base_freq = base_freq
        self.d_Embedding = d_Embedding
        self.device=device

        #the number of parameters is the number of tokens times the embedding dimention
        self.n_parameters = vocab_size * d_Embedding

    def forward(self, x):
        x = self.embedding(x)

        #intuitively the positional encoding should be multiplied, but the ML comunity sums it. maybe its just the same...
        p_encoding = positional_encoding(x.shape, self.base_freq)
        x = x+p_encoding

        x = self.dropout(x)

        return x



from einops import einsum

@torch.no_grad()
def positional_encoding(shape:torch.tensor,base_freq:float=1e-5)->torch.Tensor:
    """This function gives the positional encoding, it's slightly different then the one defined in
        the paper "Attention is all you need"
        TODO: test everything
    Args:
        shape (torch.tensor): (l,c) l=sequence lenght, c=channels
        base_freq (float, optional): The base fequency of th sinusoidal funcion.
            Defaults to 1e-5.

    Returns:
        torch.Tensor: Positional encoding with the same shape as the input (l,c)
    """
    assert len(shape) == 2, "shape must be a tuple of lenght 2, there are no batches here!"
    n_nodes, n_channels = shape
    assert n_nodes != 1 and n_channels != 1

    pos = torch.arange(0, n_channels).repeat(n_nodes, 1)


    mult = torch.logspace(0, 2, n_nodes, base=base_freq)

    elements = einsum(pos, mult, 'l c, l -> l c')

    #I don't use the cosine term because I don't really think it's very useful. I will double check in any case
    return torch.sin(elements)


class GPT2Encoder(nn.Module):

    def __init__(self, d_Embedding=768, tokenizer=Tokenizer('gpt2'), max_position_encoding=1024, dropout=0.0, device='cpu'):
        super().__init__()

        # Save the parameters
        self.tokenizer = tokenizer
        self.d_Embedding = d_Embedding
        self.max_position_encoding = max_position_encoding
        self.device = device
        self.vocab_size = tokenizer.vocab_size

        # Initialize the embedding layer
        self.embedding = nn.Embedding(tokenizer.vocab_size, d_Embedding, device=device)
        self.positional_encoding = nn.Embedding(max_position_encoding, d_Embedding, device=device)

        self.dropout = nn.Dropout(dropout)

        # Calculate the number of parameters
        self.n_parameters = tokenizer.vocab_size * \
            d_Embedding + max_position_encoding*d_Embedding

    def forward(self, x):
        #tokenize if necessary
        if type(x) == str:
            x = self.tokenizer.encode(x)

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
