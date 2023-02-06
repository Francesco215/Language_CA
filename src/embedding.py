import torch
from torch import nn
import einops

class InputEmbedding(nn.Module):
    """Layer to turn tokens into word embeddings, it also supports positional embeddings

    Args:
        nn (_type_): _description_
    """
    def __init__(self,
        hidden_dim: int = 2048,
        embedding_dim:int = 512,
        base_freq: float = 1e-5,
        dictionary_size: int = 28996, #defaults to the bert-base dictionary size
        ):
        """We first turn tokens into embedding via the self.emb function which turns each token,
        which is a scalar, into n_embedding dimentional vector.
        The input vector has size (N,L) where N is the batch size, L is the sequence lenght, which is the
        total number of tokens in the batch.
         

        Args:
            hidden_dim (int): The size of the hidden layer.
            embedding_dim (int):The size of the embedding vector.
            base_freq (float, optional): The base frequency of sinusoidal
                functions for the positional encoding. (default: 1e-4)
            dictionary_size (int, optional): The size of the dictionary of the tokenizer.
                default: 28996 which is equal to the bert-base dictionary size
        """
        super().__init__()

        self.emb=nn.Sequential(
            nn.Embedding(dictionary_size,hidden_dim),#efficent way to turn tokens into embeddings
            nn.ReLU(),
            nn.Linear(hidden_dim,embedding_dim),
            nn.ReLU()
        )
        
        self.base_freq=base_freq

    def forward(self,x):
        x=self.emb(x)
        
        p_encoding=positional_encoding(x.shape,self.base_freq)

        #intuitively the positional encoding should be multiplied, but the ML comunity sums it. maybe its just the same...
        return x+p_encoding 


@torch.no_grad()
def positional_encoding(shape:torch.tensor,base_freq:float=1e-5)->torch.Tensor:
    """This function gives the positional encoding, it's slightly different then the one defined in
        the paper "Attention is all you need"

    Args:
        shape (torch.tensor): shape=(b,l,c) or (l,c) b=batch, l=sequence lenght, c=channels
        base_freq (float, optional): The base fequency of th sinusoidal funcion. Defaults to 1e-5.

    Returns:
        torch.Tensor: Positional encoding with the same shape as the input (b,l,c) or (l,c)
    """

    rank=len(shape)
    assert rank<4
    assert rank>1

    if rank==3:
        b,l,c=shape
        pos=torch.arange(0,c).repeat(b,l,1)
    if rank==2:
        l,c=shape
        pos=torch.arange(0,c).repeat(l,1)

    assert l!=1 and c!=1
    
    #pos==torch.arange(0,c).repeat(*shape[:-1],1)

    mult=torch.logspace(0, 2, l, base=base_freq)

    elements=einops.einsum(pos, mult, '... l c, l -> ... l c')

    #I don't use the cosine term because I don't really think it's very useful. I will double check in any case
    return torch.sin(elements) 