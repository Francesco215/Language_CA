import torch
from transformers import AutoTokenizer
class linear_graph_maker:
    
    def __init__(self, window_width:int=1):
        #TODO: implement window_width

        assert type(window_width)==int, "n_nodes must be an integer"
        assert window_width>=0, "n_nodes must be non-negative"
        
        self.window_width=window_width

    @torch.no_grad()
    def __call__(self, n_nodes:int):
        """Makes a linear graph of n_nodes nodes.

        Args:
            n_nodes (int): the number of nodes.

        Returns:
            torch.Tensor: a 2xM tensor where M is the number of edges. The first row
                contains the source nodes and the second row contains the target nodes. #TODO:check this
        """
        assert type(n_nodes)==int, "n_nodes must be an integer"
        assert n_nodes>1, "n_nodes must be greater than 1"

        
        forward=[[i, i + 1] for i in range(n_nodes - 1)]
        backward=[[i + 1, i] for i in range(n_nodes - 1)]
        self_loop=[[i, i] for i in range(n_nodes)]
        edges=forward + backward + self_loop
        
        edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return edges


class random_graph_maker(linear_graph_maker):
    """makes a random graph on n_nodes nodes.
    The graph is completely random, meaning that the probability of an edge
    between two nodes is the same for all nodes.

    Args:
        n_nodes (int): the number of nodes.
        avg_n_edges (int, optional): The average number of edges per node. Defaults to 5.

    Returns:
        torch.Tensor: a 2xM tensor where M is the number of edges. The first row
            contains the source nodes and the second row contains the target nodes. #TODO:check this
    """

    def __init__(self, window_width:int=1, avg_n_edges:int=5):
        super().__init__(window_width)
        self.avg_n_edges=avg_n_edges

    @torch.no_grad()
    def __call__(self, n_nodes:int):
        #makes connection between neighbors
        edges=super().__call__(n_nodes).t()

        #this part makes non-symmetric the random edges    
        rand_edges=torch.randint(0,n_nodes,(self.avg_n_edges*n_nodes, 2))

        edges=torch.cat((edges,rand_edges),dim=0)
        edges=torch.unique(edges,dim=0).t().contiguous()

        return edges

@torch.no_grad()
def batch_graphs(nodes_list:list, edges_list:list):
    """Given a list of nodes and a list of edges, returns a batch of graphs.
    Args:
        nodes_list (list[torch.Tensor]): a list of nodes
        edges_list (list[torch.Tensor]): a list of edges

    Returns:
        tuple[torch.Tensor]: a tuple containing the nodes and the edges of the batch.
    """
    assert len(nodes_list)==len(edges_list), "nodes and edges must have the same length"

    shift=0

    for i in range(len(nodes_list)):
        edges_list[i]+=shift
        shift+=nodes_list[i].shape[0]
    
    nodes=torch.cat(nodes_list,dim=0)
    edges=torch.cat(edges_list,dim=1)

    return nodes,edges

