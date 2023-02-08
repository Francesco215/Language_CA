from src.transformerMP import TransformerBlock, attention_message

import torch, torch_geometric
import unittest


embedding_dim=100
sequence_length=130

class attention_message_test(unittest.TestCase):
    def test_attention_message(self):
        embedding_dim=17
        embedding_dim_V=21
        sequence_length=13
        n_edges=133
        heads=3

        Q= torch.rand([sequence_length,heads,embedding_dim])
        K= torch.rand([sequence_length,heads,embedding_dim])
        V= torch.rand([sequence_length,heads,embedding_dim_V])

        edge_index=torch.randint(0,sequence_length,(2,n_edges))

        senders,receivers=edge_index

        att=attention_message(K,Q,V,receivers,senders)
        self.assertEqual(V.shape,att.shape)


class transformer_test(unittest.TestCase):
    def test_tranformer_forward_type(self):
        block=TransformerBlock(embedding_dim,50,50,4)
        x=torch.rand((sequence_length,embedding_dim))
        edge_index=torch.randint(0,sequence_length,(2,sequence_length))
        out=block(x,edge_index)
        self.assertEqual(out.dtype,x.dtype)
    
    def test_transformer_forward(self):
        block=TransformerBlock(embedding_dim,50,50,4)
        x=torch.rand((sequence_length,embedding_dim))
        edge_index=torch.randint(0,sequence_length,(2,sequence_length))
        self.assertEqual(block(x,edge_index).shape,x.shape)

    