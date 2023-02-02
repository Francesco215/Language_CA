import torch, torch_geometric
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATv2Conv, Linear
from .graph_initialization import sequence_to_linear_graph, sequence_to_random_graph

class GraphCNN(nn.Module):
    """graph convolutional neural networ
    This is the simplest possible way to implement a cellular automata in graphs.
    """
    def __init__(self, num_features:int):
        """ This initializes the graph convolutional neural network.

        Args:
            num_features (int): the number of features (channels) for each node.
        """

        super().__init__()
        #print(type(num_features))
        self.conv1 = GCNConv(num_features, num_features)
        #self.conv2 = GCNConv(16, num_features)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        #x = F.relu(x)

        #x = self.conv2(x, edge_index)

        return F.relu(x)

