from src.transformerMP import TransformerBlock, attention_message

import torch
import unittest


embedding_dim=100
n_nodes=130

class attention_message_test(unittest.TestCase):
    def test_attention_message(self):
        embedding_dim=17
        embedding_dim_V=21
        n_nodes=13
        n_edges=133
        heads=3

        Q= torch.rand([n_nodes,heads,embedding_dim])
        K= torch.rand([n_nodes,heads,embedding_dim])
        V= torch.rand([n_nodes,heads,embedding_dim_V])

        edge_index=torch.randint(0,n_nodes,(2,n_edges))

        att=attention_message(K,Q,V,edge_index)
        self.assertEqual(V.shape,att.shape)

from src.transformerMP import aggregate_heads, make_heads
class transformer_test(unittest.TestCase):
    def test_head_aggregator(self):
        embedding_dim=17
        embedding_dim_V=21
        n_nodes=13
        heads=3

        head_aggregator=aggregate_heads(embedding_dim_V,embedding_dim,heads)
        x=torch.rand((n_nodes,heads,embedding_dim_V))
        out=head_aggregator(x)
        self.assertEqual(out.shape,(n_nodes,embedding_dim))

    def test_make_heads(self):
        embedding_dim=17
        embedding_dim_K=19
        embedding_dim_V=21
        n_nodes=13
        heads=3

        key=make_heads(embedding_dim,embedding_dim_K,heads)
        value=make_heads(embedding_dim,embedding_dim_V,heads)

        x=torch.rand((n_nodes,embedding_dim))
        K=key(x)
        V=value(x)

        self.assertEqual(K.shape,(n_nodes,heads,embedding_dim_K))
        self.assertEqual(V.shape,(n_nodes,heads,embedding_dim_V))

    def test_tranformer_forward_type(self):
        embedding_dim=17
        embedding_dim_K=19
        embedding_dim_V=21
        n_nodes=13
        n_edges=133
        heads=3

        block=TransformerBlock(embedding_dim,dK=embedding_dim_K,dV=embedding_dim_V,heads=heads)
        x=torch.rand((n_nodes,embedding_dim))
        edge_index=torch.randint(0,n_nodes,(2,n_edges))
        out=block(x,edge_index)
        self.assertEqual(out.dtype,x.dtype)
    
    def test_transformer_forward(self):
        embedding_dim=17
        embedding_dim_K=19
        embedding_dim_V=21
        n_nodes=13
        n_edges=133
        heads=3

        block=TransformerBlock(embedding_dim,dK=embedding_dim_K,dV=embedding_dim_V,heads=heads)
        x=torch.rand((n_nodes,embedding_dim))
        edge_index=torch.randint(0,n_nodes,(2,n_edges))
        out=block(x,edge_index)

        self.assertEqual(out.shape,x.shape)

    