
import torch, torch_geometric
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATv2Conv, Linear
import einops

from .embedding import InputEmbedding



class GraphAttentionNetwork(nn.Module):
    """This graph attention network is based on the GATv2Conv from the torch_geometric library.
    
    """
    def __init__(self, embedding=InputEmbedding()
        ):
        
        super().__init__()
        
        self.embedding = embedding

        #self.conv = GATv2Conv(num_features, num_features, heads=4, concat=True, share_weights=True)
        #self.conv2 = GCNConv(16, num_features)


    def forward(self, data):
        pass



