import unittest
import torch

from src.transformerMP import normalize_strength

class select_index_test(unittest.TestCase):
    def test_select_index(self):
        x=torch.rand((10,2))
        index=torch.randint(0,10,(100,))
        self.assertEqual(x[index].shape,(100,2))
        
    def test_select_index_2(self):
        x=torch.rand((10,2))
        index=torch.randint(0,10,(100,40))
        self.assertEqual(x[index].shape,(100,40,2))

    def test_select_index_3(self):
        x=torch.rand((10,8,2))
        index=torch.randint(0,10,(100,))
        self.assertEqual(x[index].shape,(100,8,2))

    def test_index_add(self):
        receivers = torch.tensor([1, 2, 1, 0])
        strength = torch.tensor([0.5, 0.2, 0.3, 0.7])
        nodes=torch.zeros(3)
        nodes=nodes.index_add(-1,receivers,strength)

        assertion=(nodes==torch.tensor([0.7000, 0.8000, 0.2000])).all()

        self.assertTrue(assertion)

    def test_index_add_with_heads(self):
        senders = torch.tensor([0, 1, 2, 2])
        receivers = torch.tensor([1, 2, 1, 0])
        strength = torch.tensor([[0.5, 0.2, 0.3, 0.7],[0.3, 0.2, 0.3, 0.7]]).t()

        nodes=torch.zeros([3,2])
        
        nodes=nodes.index_add(0,receivers,strength)

        desired_nodes=torch.tensor([[0.7000, 0.7000],
                                    [0.8000, 0.6000],
                                    [0.2000, 0.2000]])
        
        assertion=(nodes==desired_nodes).all()
        self.assertTrue(assertion)


        strength=strength/nodes[receivers]
        #the first and the third sum to one, they both point at the node number 1
        desired_strength=torch.tensor([[0.6250, 0.5000], 
                                       [1.0000, 1.0000],
                                       [0.3750, 0.5000],
                                       [1.0000, 1.0000]])

        assertion=(strength==desired_strength).all()
        self.assertTrue(assertion)

    def test_shape_index_add_with_heads_random(self):
        sequence_length=74
        n_edges=563
        heads=6
        receivers=torch.randint(0,sequence_length,(n_edges,))
        strength = torch.rand([n_edges,heads])

        nodes=torch.zeros([sequence_length,heads])
        
        nodes=nodes.index_add(0,receivers,strength)
        self.assertEqual(nodes.shape,(sequence_length,heads))

        strength=strength/nodes[receivers]
        self.assertEqual(strength.shape,(n_edges,heads))

    def test_shape_normalize_strength(self):
        n_nodes=74
        n_edges=563
        heads=6
        receivers=torch.randint(0,n_nodes,(n_edges,))
        strength = torch.rand([n_edges,heads])

        nodes=torch.zeros([n_nodes,heads])
        
        strength=normalize_strength(strength,receivers,n_nodes,heads)
        self.assertEqual(strength.shape,(n_edges,heads))

    def test_normalize_strength(self):
        n_nodes=74
        n_edges=563
        heads=6
        receivers=torch.randint(0,n_nodes,(n_edges,))
        strength = torch.rand([n_edges,heads])
        
        strength=normalize_strength(strength,receivers,n_nodes,heads)
        
        nodes=torch.zeros([n_nodes,heads])
        #after this commit where i wrote this comment, for some reason using index_add inplace gives an error 🤔?
        nodes=nodes.index_add(0,receivers,strength)

        assertion=torch.allclose(nodes,torch.ones_like(nodes))
        self.assertTrue(assertion)

from src.graph_initialization import random_graph_maker
class max_index_test(unittest.TestCase):
    #if this test fails, the softmax used won't work properly
    def test_max_reduction(self):
        sequence_length=23
        heads=7
        graph_maker=random_graph_maker(2,5)
        senders,receivers=graph_maker(sequence_length)
        attention=torch.randn((receivers.shape[0],heads))

        new_attention_baseline=torch.empty_like(attention)
        for i in range(sequence_length):
            idx=receivers==i
            new_attention_baseline[idx]=attention[idx]-attention[idx].max(dim=0).values


        translation=torch.zeros(sequence_length,heads)

        translation=translation.scatter_reduce(0, receivers.repeat(heads,1).t(), attention, reduce='amax', include_self=False)

        new_attention=attention-translation[receivers]

        self.assertTrue(torch.allclose(new_attention,new_attention_baseline,1e-3,1e-3))
        