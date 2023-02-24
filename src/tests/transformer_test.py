from src.transformerMP import attention_message


import torch
import unittest


class attention_message_test(unittest.TestCase):
    def test_attention_message(self):
        embedding_dim=17
        embedding_dim_V=21
        n_nodes=13
        n_edges=133
        heads=3

        Q= torch.randn([n_nodes,heads,embedding_dim])
        K= torch.randn([n_nodes,heads,embedding_dim])
        V= torch.randn([n_nodes,heads,embedding_dim_V])

        edge_index=torch.randint(0,n_nodes,(2,n_edges))

        att,_=attention_message(Q,K,V,edge_index)
        self.assertEqual(V.shape,att.shape)
        self.assertFalse(att.isnan().any())
        self.assertFalse(att.isinf().any())

    def test_attention_message_high_variance(self):
        embedding_dim=17
        embedding_dim_V=21
        n_nodes=13
        n_edges=133
        heads=3

        Q = torch.randn([n_nodes,heads,embedding_dim])*40
        K = torch.randn([n_nodes,heads,embedding_dim])*70
        V = torch.randn([n_nodes,heads,embedding_dim_V])*90

        edge_index=torch.randint(0,n_nodes,(2,n_edges))

        att,_=attention_message(Q,K,V,edge_index)
        self.assertEqual(V.shape,att.shape)
        self.assertFalse(att.isnan().any())
        self.assertFalse(att.isinf().any())


from src.transformerMP import AttentionBlock, aggregate_heads, make_QKV
class transformer_test(unittest.TestCase):
    def test_head_aggregator(self):
        embedding_dim=17
        embedding_dim_V=21
        n_nodes=13
        heads=3

        head_aggregator=aggregate_heads(embedding_dim_V,embedding_dim,heads)
        x=torch.randn((n_nodes,heads,embedding_dim_V))
        out=head_aggregator(x)
        self.assertEqual(out.shape,(n_nodes,embedding_dim))

    def test_make_heads(self):
        embedding_dim=17
        embedding_dim_K=19
        embedding_dim_V=21
        n_nodes=13
        heads=3

        QKV_maker=make_QKV(embedding_dim,embedding_dim_K,embedding_dim_V,heads)

        x=torch.randn((n_nodes,embedding_dim))
        Q,K,V=QKV_maker(x)

        self.assertEqual(K.shape,(n_nodes,heads,embedding_dim_K))
        self.assertEqual(V.shape,(n_nodes,heads,embedding_dim_V))

    def test_tranformer_forward_type(self):
        embedding_dim=17
        embedding_dim_K=19
        embedding_dim_V=21
        n_nodes=13
        n_edges=133
        heads=3

        block=AttentionBlock(embedding_dim,dK=embedding_dim_K,dV=embedding_dim_V,heads=heads)
        x=torch.randn((n_nodes,embedding_dim))
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

        block=AttentionBlock(embedding_dim,dK=embedding_dim_K,dV=embedding_dim_V,heads=heads)
        x=torch.randn((n_nodes,embedding_dim))
        edge_index=torch.randint(0,n_nodes,(2,n_edges))
        out=block(x,edge_index)

        self.assertEqual(out.shape,x.shape)
        self.assertFalse(out.isnan().any())
        self.assertFalse(out.isinf().any())

    
from src.GPT2 import AttentionBlockGPT2, transform_heads, interact_heads

class GPT2_transformer_test(unittest.TestCase):
    def test_transform_heads(self):
        embedding_dim=24
        embedding_dim_V=21
        n_nodes=13
        heads=3

        value = transform_heads(embedding_dim, embedding_dim_V, heads)
        x=torch.randn((n_nodes,embedding_dim))
        out=value(x)
        self.assertEqual(out.shape,(n_nodes,heads,embedding_dim_V//heads))
        self.assertFalse(out.isnan().any())
        self.assertFalse(out.isinf().any())

    def test_interact_heads(self):
        embedding_dim=24
        embedding_dim_V=21
        n_nodes=13
        heads=3

        value = interact_heads(embedding_dim_V, embedding_dim)
        x=torch.randn((n_nodes,heads,embedding_dim_V//heads))
        out=value(x)
        self.assertEqual(out.shape,(n_nodes,embedding_dim))
        self.assertFalse(out.isnan().any())
        self.assertFalse(out.isinf().any())


    def test_GPT2_transformer_forward_type(self):
        embedding_dim=24
        embedding_dim_K=18
        embedding_dim_V=21
        n_nodes=13
        n_edges=133
        heads=3

        block=AttentionBlockGPT2(embedding_dim,dK=embedding_dim_K,dV=embedding_dim_V,heads=heads)
        x=torch.randn((n_nodes,embedding_dim))
        edge_index=torch.randint(0,n_nodes,(2,n_edges))
        out=block(x,edge_index)
        self.assertEqual(out.dtype,x.dtype)
        self.assertFalse(out.isnan().any())
        self.assertFalse(out.isinf().any())

    def test_GPT2_transformer_forward(self):
        embedding_dim=24
        embedding_dim_K=18
        embedding_dim_V=21
        n_nodes=13
        n_edges=133
        heads=3

        block=AttentionBlockGPT2(embedding_dim,dK=embedding_dim_K,dV=embedding_dim_V,heads=heads)
        x=torch.randn((n_nodes,embedding_dim))
        edge_index=torch.randint(0,n_nodes,(2,n_edges))
        out=block(x,edge_index)

        self.assertEqual(out.shape,x.shape)
        self.assertFalse(out.isnan().any())
        self.assertFalse(out.isinf().any())