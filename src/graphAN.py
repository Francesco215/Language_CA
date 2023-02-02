
import torch, torch_geometric
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATv2Conv, Linear
import einops




class GraphAttentionNetwork(nn.Module):
    """This graph attention network is based on the GATv2Conv from the torch_geometric library.
    
    """
    def __init__(self, num_features):
        """ This initializes the graph convolutional neural network.

        Args:
            num_features (int): the number of features (channels) for each node.
        """
        super().__init__()
        pass

        #self.conv = GATv2Conv(num_features, num_features, heads=4, concat=True, share_weights=True)
        #self.conv2 = GCNConv(16, num_features)


    def forward(self, data):
        pass




class InputEmbedding(nn.Module):
    """Layer to turn tokens into word embeddings, it also supports positional embeddings

    Args:
        nn (_type_): _description_
    """
    def __init__(self,
        hidden_dim: int = 2048,
        embedding_dim:int = 512,
        base_freq: float = 1e-5,
        ):
        """We first turn tokens into embedding via the self.emb function which turns each token,
        which is a scalar, into n_embedding dimentional vector.
        The input vector has size (N,C,L) where N is the batch size, C is the number of channels
        which is equal to 1 in the case of tokens, and L is the sequence lenght, which is the
        total number of tokens

        Args:
            hidden_dim (int): _description_
            embedding_dim (int): _description_
            base_freq (float, optional): The base frequency of sinusoidal
                functions for the positional encoding. (default: 1e-4)
        """
        super().__init__()

        self.emb=nn.Sequential(
            nn.Conv1d(1,hidden_dim,1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim,embedding_dim,1),
            nn.ReLU()
        )
        
        self.base_freq=base_freq

    def forward(self,x):
        


        #See if it the rearrangments slow down computation and if they can be avoided all together
        #As of now they are needed because the Conv1d operation wants (b,c,l) tensor instead of (b,l,c)
        x=einops.rearrange(x,'... l -> ... 1 l')
        x=self.emb(x)
        x=einops.rearrange(x,'... c l -> ... l c')
        
        p_encoding=positional_encoding(x.shape,self.base_freq)

        #intuitively the positional encoding should be multiplied, but the ML comunity sums it. maybe its just the same...
        return x+p_encoding 



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