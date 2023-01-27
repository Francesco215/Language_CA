import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from graphCA import sequence_to_linear_graph, sequence_to_random_graph

class GCNN(nn.Module):
    """graph convolutional neural networ
    This is the simplest possible way to implement a cellular automata in graphs.
    """
    def __init__(self, num_features):
        """ This initializes the graph convolutional neural network.

        Args:
            num_features (int): the number of features (channels) for each node.
        """
        super().__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)




