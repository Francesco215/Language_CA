import unittest

import torch

from src.graph_initialization import linear_graph_maker, random_graph_maker, batch_graphs
class graph_init_test(unittest.TestCase):
    def test_linear_graph(self):
        n_nodes=4
        graph_maker=linear_graph_maker(window_width=1)

        edges=graph_maker(n_nodes)

        expected_n_edges=n_nodes*3-2

        self.assertEqual(edges.shape,(2,expected_n_edges))

        expected_output= torch.tensor([[0, 1, 2, 1, 2, 3, 0, 1, 2, 3],
                                       [1, 2, 3, 0, 1, 2, 0, 1, 2, 3]])

        assertion=(edges==expected_output).all()
        self.assertTrue(assertion)


    def test_random_graph(self):
        n_nodes=56
        average_n_edges=5

        graph_maker=random_graph_maker(window_width=1,avg_n_edges=average_n_edges)
        edges=graph_maker(n_nodes)

        self.assertEqual(edges.shape[0],2)

    
    def test_torch_unique(self):
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6], [7, 8, 9]])
        t_expeced=torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        assertion=(torch.unique(t, dim=0)==t_expeced).all()

        self.assertTrue(assertion)

    def test_batch_graphs(self):
        nodes_list=[torch.rand((2,2)), torch.rand((3,2)), torch.rand((4,2))]
        edges_list=[torch.tensor([[0,1],[1,0]]), torch.tensor([[0,1,2],[1,2,0]]), torch.tensor([[0,1,2,3],[1,2,3,0]])]

        nodes,edges=batch_graphs(nodes_list,edges_list)

        self.assertEqual(nodes.shape[0],9)
        self.assertEqual(edges.shape[1],9)

        self.assertEqual(nodes.shape[1],2)
        self.assertEqual(edges.shape[0],2)

    