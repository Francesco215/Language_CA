from src.decoder import Loss
from src.graphAN import GraphAttentionNetwork
from src import Encoder, Decoder, TransformerBlock

import torch,unittest

from src.data_loader import Tokenizer, Wiki
from src.graph_initialization import batch_graphs, random_graph_maker

class Back_to_BackTest(unittest.TestCase):

    def test_flow(self):
        hidden_dim=200
        embedding_dim=50
        dK=50
        dV=50
        heads=4
        tokenizer=Tokenizer()

        vocab_size=tokenizer.vocab_size

        encoder=Encoder(hidden_dim,embedding_dim,vocab_size)
        transformer=TransformerBlock
        decoder=Decoder(hidden_dim,embedding_dim,vocab_size)

        network=GraphAttentionNetwork(tokenizer,encoder,decoder,transformer,2,dK,dV,heads)

        n_nodes=40
        n_edges=302
        nodes=torch.randint(0,vocab_size,(n_nodes,))
        edge_index=torch.randint(0,n_nodes,(2,n_edges))
        out=network(nodes,edge_index)
        out=decoder(out)
        self.assertEqual(out.shape,(n_nodes,vocab_size))

    def test_inference(self):  
        hidden_dim=200
        embedding_dim=50
        dK=50
        dV=50
        heads=4
        tokenizer=Tokenizer()

        vocab_size=tokenizer.vocab_size

        encoder=Encoder(hidden_dim,embedding_dim,vocab_size)
        transformer=TransformerBlock
        decoder=Decoder(hidden_dim,embedding_dim,vocab_size)

        network=GraphAttentionNetwork(tokenizer,encoder,decoder,transformer,2,dK,dV,heads)

        n_nodes=40
        n_edges=302
        nodes=torch.randint(0,vocab_size,(n_nodes,))
        edge_index=torch.randint(0,n_nodes,(2,n_edges))
        out=network.inference(nodes,edge_index)
        self.assertEqual(type(out),str)

    def test_loss(self):
        hidden_dim=200
        embedding_dim=50
        dK=50
        dV=50
        heads=4
        tokenizer=Tokenizer()

        vocab_size=tokenizer.vocab_size

        encoder=Encoder(hidden_dim,embedding_dim,vocab_size)
        transformer=TransformerBlock
        decoder=Decoder(hidden_dim,embedding_dim,vocab_size)
        loss=Loss(decoder)

        network=GraphAttentionNetwork(tokenizer,encoder,decoder,transformer,2,dK,dV,heads)

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

        
        hidden_dim=200
        embedding_dim=50
        dK=50
        dV=50
        heads=4
        tokenizer=Tokenizer()

        vocab_size=tokenizer.vocab_size

        encoder=Encoder(hidden_dim,embedding_dim,vocab_size)
        transformer=TransformerBlock
        decoder=Decoder(hidden_dim,embedding_dim,vocab_size)

        network=GraphAttentionNetwork(tokenizer,encoder,decoder,transformer,2,dK,dV,heads)

        nodes,edge_index=data[0]
        n_nodes=len(nodes)
        out=network(nodes,edge_index)
        out=decoder(out)
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

        
        hidden_dim=200
        embedding_dim=50
        dK=50
        dV=50
        heads=4
        tokenizer=Tokenizer()

        vocab_size=tokenizer.vocab_size

        encoder=Encoder(hidden_dim,embedding_dim,vocab_size)
        transformer=TransformerBlock
        decoder=Decoder(hidden_dim,embedding_dim,vocab_size)

        network=GraphAttentionNetwork(tokenizer,encoder,decoder,transformer,2,dK,dV,heads)

        nodes,edge_index=data[0:2]
        nodes,edge_index=batch_graphs(nodes,edge_index)
        n_nodes=len(nodes)
        out=network(nodes,edge_index)
        out=decoder(out)
        self.assertEqual(out.shape,(n_nodes,vocab_size))
