from src import Encoder, Decoder, AttentionBlock, Tokenizer, random_graph_maker, Wiki, GraphAttentionNetwork, Loss

import torch,unittest

class Back_to_BackTest(unittest.TestCase):
    def test_few_training_steps(self):
        device='cpu'
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
        transformer_layers=2
        tokenizer=Tokenizer()

        vocab_size=tokenizer.vocab_size

        encoder=Encoder(hidden_dim,embedding_dim,vocab_size)
        transformer=AttentionBlock
        decoder=Decoder(hidden_dim,embedding_dim,vocab_size)

        model=GraphAttentionNetwork(tokenizer,encoder,decoder,transformer,transformer_layers,dK,dV,heads)

        loss_function=Loss(decoder)
        lr=1e-2
        gamma=0.99

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

        n_epochs=2
        model.train()
        losses=[]
        for _ in range(n_epochs):
            for page in data:
                nodes,edge_index=page
                nodes.to(device)
                edge_index.to(device)
                optimizer.zero_grad()  # reinitialize the gradient to zero
                
                prediction=model(nodes,edge_index)
                
                loss=loss_function(prediction,nodes)
                losses.append(loss.item())
                loss.backward()

                optimizer.step()

        self.assertEqual(len(losses),len(data)*n_epochs)