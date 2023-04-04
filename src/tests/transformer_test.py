import einops
import torch
import unittest
from src.attention import AttentionMessage, overlaps
from .test_utils import og_attention_message
class attention_message_utils(unittest.TestCase):
    def og_attention_test(self):
        embedding_dim=17
        embedding_dim_V=21
        n_nodes=13
        n_edges=133
        heads=3


        mh_att=torch.nn.MultiheadAttention(embedding_dim,heads)
class attention_message_test(unittest.TestCase):
    def test_attention_message(self):
        embedding_dim = 17
        embedding_dim_V = 21
        n_nodes = 13
        n_edges = 133
        heads = 3

        Q = torch.randn([n_nodes, heads, embedding_dim])
        K = torch.randn([n_nodes, heads, embedding_dim])
        V = torch.randn([n_nodes, heads, embedding_dim_V])

        Q1 = Q.clone()

        Q.requires_grad = True
        Q.retain_grad()


        #edge_index=torch.randint(0, input_size, (2, 10), device=device).unique(dim=1)
        edge_index=torch.randint(0,n_nodes,(2,n_edges)).unique(dim=1)
        attention_message=AttentionMessage()
        x, att = attention_message(Q, K, V, edge_index)
        x = x.mean()
        x.backward()


        Q1.requires_grad = True
        Q1.retain_grad()


        x1, att1 = og_attention_message(Q1, K, V, edge_index)
        x1 = x1.mean()
        x1.backward()

        senders,receivers=edge_index
        self.assertTrue(torch.isclose(x,x1,1e-3,1e-3).all())

    def test_attention_message_high_variance(self):
        embedding_dim=17
        embedding_dim_V=21
        n_nodes=13
        n_edges=133
        heads=3

        Q = torch.randn([n_nodes,heads,embedding_dim])*40
        K = torch.randn([n_nodes,heads,embedding_dim])*70
        V = torch.randn([n_nodes,heads,embedding_dim_V])*90

        edge_index=torch.randint(0,n_nodes,(2,n_edges)).unique(dim=1)
        attention_message = AttentionMessage()

        att,_=attention_message(Q,K,V,edge_index)
        self.assertEqual(V.shape,att.shape)
        self.assertFalse(att.isnan().any())
        self.assertFalse(att.isinf().any())


class attention_message_gradient_test(unittest.TestCase):

    def test_attention_message_Q_gradient(self):
        embedding_dim = 17
        embedding_dim_V = 21
        n_nodes = 13
        n_edges = 133
        heads = 3
        attention_message=AttentionMessage()


        Q = torch.randn([n_nodes, heads, embedding_dim])
        K = torch.randn([n_nodes, heads, embedding_dim])
        V = torch.randn([n_nodes, heads, embedding_dim_V])
        Q1 = Q.clone()
        edge_index = torch.randint(0, n_nodes, (2, n_edges)).unique(dim=1)


        Q.requires_grad=True
        Q.retain_grad()
        x, _ = attention_message(Q, K, V, edge_index)
        x=x.mean()
        x.backward()

        self.assertFalse(Q.grad.isnan().any())
        self.assertFalse(Q.grad.isinf().any())


        
        Q1.requires_grad=True
        Q1.retain_grad()
        x1,_=og_attention_message(Q1, K, V, edge_index)
        x1=x1.mean()
        x1.backward()


        self.assertFalse(Q1.grad.isnan().any())
        self.assertFalse(Q1.grad.isinf().any())

        self.assertTrue(torch.allclose(Q.grad, Q1.grad, 1e-3, 1e-3))

    def test_attention_message_K_gradient(self):
        embedding_dim = 17
        embedding_dim_V = 21
        n_nodes = 13
        n_edges = 133
        heads = 3
        attention_message=AttentionMessage()


        Q = torch.randn([n_nodes, heads, embedding_dim])
        K = torch.randn([n_nodes, heads, embedding_dim])
        V = torch.randn([n_nodes, heads, embedding_dim_V])
        K_1 = K.clone()
        edge_index = torch.randint(0, n_nodes, (2, n_edges)).unique(dim=1)


        K.requires_grad=True
        K.retain_grad()
        x, _ = attention_message(Q, K, V, edge_index)
        x=x.mean()
        x.backward()

        self.assertFalse(K.grad.isnan().any())
        self.assertFalse(K.grad.isinf().any())

        
        K_1.requires_grad=True
        K_1.retain_grad()
        x,_=og_attention_message(Q, K_1, V, edge_index)
        x=x.mean()
        x.backward()


        self.assertFalse(K_1.grad.isnan().any())
        self.assertFalse(K_1.grad.isinf().any())

        self.assertTrue(torch.allclose(K.grad, K_1.grad, 1e-3, 1e-3))

    def test_attention_message_V_gradient(self):
        embedding_dim = 17
        embedding_dim_V = 21
        n_nodes = 13
        n_edges = 133
        heads = 3
        attention_message=AttentionMessage()


        Q = torch.randn([n_nodes, heads, embedding_dim])
        K = torch.randn([n_nodes, heads, embedding_dim])
        V = torch.randn([n_nodes, heads, embedding_dim_V])
        V_1 = V.clone()
        edge_index = torch.randint(0, n_nodes, (2, n_edges))


        V.requires_grad=True
        V.retain_grad()
        x, _ = og_attention_message(Q, K, V, edge_index)
        x=x.mean()
        x.backward()

        self.assertFalse(V.grad.isnan().any())
        self.assertFalse(V.grad.isinf().any())

        
        V_1.requires_grad=True
        V_1.retain_grad()
        x,_=attention_message(Q, K, V_1, edge_index)
        x=x.mean()
        x.backward()


        self.assertFalse(V_1.grad.isnan().any())
        self.assertFalse(V_1.grad.isinf().any())

        self.assertTrue(torch.allclose(V.grad, V_1.grad, 1e-3, 1e-3))


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
        embedding_dim_K=18
        embedding_dim_V=21
        n_nodes=13
        heads=3

        QKV_maker=make_QKV(embedding_dim,embedding_dim_K,embedding_dim_V,heads, True)

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

    
from src.GPT2 import AttentionBlockGPT2

class GPT2_transformer_test(unittest.TestCase):
    """def test_transform_heads(self):
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
        self.assertFalse(out.isinf().any())"""

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

    