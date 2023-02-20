import torch
from torch import nn

class Encoder(nn.Module):
    """Layer to turn tokens into word embeddings, it also supports positional embeddings

    Args:
        nn (_type_): _description_
    """
    def __init__(self,
        embedding_dim:int = 512,
        base_freq: float = 1e-5,
        vocab_size: int = 28996, #defaults to the bert-base dictionary size,
        dropout:float=0.1
        ):
        """We first turn tokens into embedding via the self.emb function which turns each token,
        which is a scalar, into n_embedding dimentional vector.
        The input vector has size (N,L) where N is the batch size, L is the sequence lenght, which is the
        total number of tokens in the batch.
         

        Args:
            embedding_dim (int):The size of the embedding vector.
            base_freq (float, optional): The base frequency of sinusoidal
                functions for the positional encoding. (default: 1e-4)
            vocab_size (int, optional): The size of the dictionary of the tokenizer.
                default: 28996 which is equal to the bert-base dictionary size
            dropout (float, optional): The dropout rate. Defaults to 0.1.
                TODO: chec if it should be applied to the positional encoding.
        """
        super().__init__()


        #the embedding layer turns each token into a vector of size embedding_dim
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.dropout=nn.Dropout(dropout)       


        self.vocab_size=vocab_size
        self.base_freq=base_freq
        self.embedding_dim=embedding_dim

        #the number of parameters is the number of tokens times the embedding dimention
        self.n_parameters= vocab_size * embedding_dim


    def forward(self,x):
        x=self.embedding(x)
        
        #intuitively the positional encoding should be multiplied, but the ML comunity sums it. maybe its just the same...
        p_encoding=positional_encoding(x.shape,self.base_freq)
        x=x+p_encoding

        x=self.dropout(x)
        
        return x



import einops

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