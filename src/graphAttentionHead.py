from torch import nn
from torch_geometric.nn import GCNConv, GATv2Conv




class GraphAttetionHead(nn.Module):

    def __init__(self, num_features, num_classes, num_heads=4, dropout=0.5):
        super(GraphAttetionHead, self).__init__()
        self.conv = GATv2Conv(num_features, num_features, heads=num_heads, concat=True, share_weights=True)
        self.conv2 = GCNConv(num_features * num_heads, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)