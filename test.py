import torch, torch_geometric
from src import graphCA
import networkx as nx

sequence=torch.rand(10)

print(sequence)

data=graphCA.sequence_to_random_graph(sequence,3)

g = torch_geometric.utils.to_networkx(data, to_undirected=True)
nx.draw(g)