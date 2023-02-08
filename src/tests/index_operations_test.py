import unittest
import torch

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

    def test_index_add_second_method(self):
        senders = torch.tensor([0, 1, 2, 2])
        receivers = torch.tensor([1, 2, 1, 0])
        strength = torch.tensor([0.5, 0.2, 0.3, 0.7])

        nodes=torch.zeros(3)

        nodes[receivers]+=strength

        assertion=(nodes==torch.tensor([0.7000, 0.8000, 0.2000])).all()

        self.assertTrue(assertion)