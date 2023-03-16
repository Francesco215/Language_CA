from src import Encoder, Decoder, AttentionBlock, Tokenizer, batch_graphs, random_graph_maker, linear_unidirectional_graph_maker, Wiki, GraphAttentionNetwork, Loss, BlockGenerator

import torch,unittest

from src.encoder import GPT2Encoder
from src.decoder import GPT2Decoder
class Back_to_BackTest(unittest.TestCase):

    def test_flow(self):
        embedding_dim=50
        dK=50
        dV=50
        heads=4
        transformer_layers=2
        tokenizer=Tokenizer("bert-base-cased",max_length=50)

        vocab_size=tokenizer.vocab_size

        encoder=Encoder(embedding_dim,vocab_size)
        block_generator = BlockGenerator(AttentionBlock, embedding_dim, dK, dV, heads, rotary_encoding=True)

        network=GraphAttentionNetwork(tokenizer,encoder,block_generator,n_blocks=transformer_layers)

        n_nodes=40
        n_edges=302
        nodes=torch.randint(0,vocab_size,(n_nodes,))
        edge_index=torch.randint(0,n_nodes,(2,n_edges))
        out=network(nodes,edge_index)
        out=network.decoder(out)
        self.assertEqual(out.shape,(n_nodes,vocab_size))

    def test_inference(self):  
        embedding_dim = 50
        dK = 50
        dV = 50
        heads = 4
        transformer_layers = 2
        tokenizer = Tokenizer("bert-base-cased", max_length=50)

        vocab_size = tokenizer.vocab_size

        encoder = Encoder(embedding_dim, vocab_size)
        block_generator = BlockGenerator(AttentionBlock, embedding_dim, dK, dV, heads)

        network = GraphAttentionNetwork(tokenizer, encoder, block_generator, n_blocks=transformer_layers)

        n_nodes=40
        n_edges=302
        nodes=torch.randint(0,vocab_size,(n_nodes,))
        edge_index=torch.randint(0,n_nodes,(2,n_edges))
        out=network.inference(nodes,edge_index)
        self.assertEqual(type(out),str)

    def test_loss(self):
        embedding_dim = 50
        dK = 50
        dV = 50
        heads = 4
        transformer_layers = 2
        tokenizer = Tokenizer("bert-base-cased", max_length=50)

        vocab_size = tokenizer.vocab_size

        encoder = Encoder(embedding_dim, vocab_size)
        block_generator = BlockGenerator(AttentionBlock, embedding_dim, dK, dV, heads)

        network = GraphAttentionNetwork(tokenizer, encoder, block_generator, n_blocks=transformer_layers)
        loss=Loss(network.decoder)

        n_nodes=40
        n_edges=302
        nodes=torch.randint(0,vocab_size,(n_nodes,))
        edge_index=torch.randint(0,n_nodes,(2,n_edges))
        out=network(nodes,edge_index)

        y=torch.randint(0,vocab_size,(n_nodes,))
        l=loss(out,y)

        self.assertEqual(l.shape, torch.Size([]))

class Back_to_BackTest_from_dataset(unittest.TestCase):

    def test_flow_single_datapoint(self):
        dummy_dataset=[
            {'text':"This is a test"},
            {'text':"This is another test but with different length"},
            {'text':"This is the last test string, i promise no more strings"}
        ]

        tokenizer=Tokenizer(max_length=50)
        graph_maker=random_graph_maker(1,5)
        data=Wiki(dummy_dataset,tokenizer,graph_maker)

        
        embedding_dim=50
        dK=50
        dV=50
        heads=4
        transformer_layers=2

        vocab_size=tokenizer.vocab_size

        encoder=Encoder(embedding_dim,vocab_size)
        block_generator = BlockGenerator(AttentionBlock, embedding_dim, dK, dV, heads)

        network=GraphAttentionNetwork(tokenizer,encoder,block_generator,n_blocks=transformer_layers)

        nodes,edge_index=data[0]
        n_nodes=len(nodes)
        out=network(nodes,edge_index)
        out=network.decoder(out)
        self.assertEqual(out.shape,(n_nodes,vocab_size))

    def test_flow_single_batched_datapoint(self):  
        dummy_dataset=[
            {'text':"This is a test"},
            {'text':"This is another test but with different length"},
            {'text':"This is the last test string, i promise no more strings"}
        ]

        tokenizer=Tokenizer(max_length=50)
        graph_maker=random_graph_maker(1,5)
        data=Wiki(dummy_dataset,tokenizer,graph_maker)

        
        embedding_dim=50
        dK=50
        dV=50
        heads=4
        transformer_layers=2

        vocab_size=tokenizer.vocab_size

        encoder=Encoder(embedding_dim,vocab_size)
        block_generator = BlockGenerator(AttentionBlock, embedding_dim, dK, dV, heads)

        network=GraphAttentionNetwork(tokenizer,encoder,block_generator,n_blocks=transformer_layers)

        nodes,edge_index=data[0:2]
        nodes,edge_index=batch_graphs(nodes,edge_index)
        n_nodes=len(nodes)
        out=network(nodes,edge_index)
        out=network.decoder(out)
        self.assertEqual(out.shape,(n_nodes,vocab_size))

from src.GPT2 import GPT2, GPT2_Block

class GPT2BacktoBack(unittest.TestCase):
    def test_flow(self):
        embedding_dim=60
        dK=52
        dV=64
        heads=4
        intermediate_size=50
        dropout=0.1
        tokenizer=Tokenizer('gpt2')

        vocab_size=tokenizer.vocab_size

        encoder=GPT2Encoder(embedding_dim,tokenizer)
        decoder=GPT2Decoder(embedding_dim,tokenizer)
        block_generator=BlockGenerator(GPT2_Block,embedding_dim, dK, dV, heads, intermediate_size, dropout)
        network= GPT2(tokenizer,encoder,block_generator,decoder,12)

        n_nodes=40
        n_edges=302
        nodes=torch.randint(0,vocab_size,(n_nodes,))
        edge_index=torch.randint(0,n_nodes,(2,n_edges))
        out=network.calculate_final_embedding(nodes,edge_index)
        self.assertEqual(out.shape,(n_nodes,vocab_size))

    def test_flow_single_batched_datapoint(self):  
        embedding_dim=60
        dK=52
        dV=64
        heads=4
        intermediate_size=50
        dropout=0.1
        tokenizer=Tokenizer('gpt2')

        dummy_dataset=[
            {'text':"This is a test"},
            {'text':"This is another test but with different length"},
            {'text':"This is the last test string, i promise no more strings"}
        ]
        graph_maker=linear_unidirectional_graph_maker(5)
        data=Wiki(dummy_dataset,tokenizer,graph_maker)


        vocab_size=tokenizer.vocab_size

        encoder = GPT2Encoder(embedding_dim, tokenizer)
        decoder = GPT2Decoder(embedding_dim, tokenizer)
        block_generator = BlockGenerator(GPT2_Block, embedding_dim, dK, dV, heads, intermediate_size, dropout)
        network = GPT2(tokenizer, encoder, block_generator, decoder, 12)

        nodes,edge_index=data[0:2]
        nodes,edge_index=batch_graphs(nodes,edge_index)
        n_nodes,n_edges=len(nodes),len(edge_index[0])

        out=network.calculate_final_embedding(nodes,edge_index)
        self.assertEqual(out.shape,(n_nodes,vocab_size))
