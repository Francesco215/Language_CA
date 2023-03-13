import torch, unittest
from src import Tokenizer

from torch import nn


from src.GPT2 import GPT2, GPT2_Block, Conv1D_to_Linear
from src import linear_unidirectional_graph_maker
from transformers.modeling_utils import Conv1D
from src.encoder import GPT2Encoder
from src.decoder import GPT2Decoder

from src.graphAN import BlockGenerator

class GPT2_loading_functions(unittest.TestCase):
    def test_Conv1D_to_Linear(self):
        conv=Conv1D(60,40)
        lin=nn.Linear(40,60)

        Conv1D_to_Linear(conv,lin)

        x=torch.rand(17,40)
        self.assertTrue(torch.allclose(conv(x),lin(x)))
        conv=Conv1D(60,40)
        lin=nn.Linear(40,60)

        Conv1D_to_Linear(conv,lin)

        x=torch.rand(17,40)
        self.assertTrue(torch.allclose(conv(x),lin(x)))

    def test_Conv1D_to_Linear_error(self):

        conv=Conv1D(40,60)
        lin=nn.Linear(40,60)

        self.assertRaises(AssertionError, Conv1D_to_Linear, conv, lin)


    def test_loading(self):
        tokenizer = Tokenizer('gpt2')

        encoder=GPT2Encoder()
        decoder=GPT2Decoder()
        block_generator = BlockGenerator(GPT2_Block)
        model = GPT2(tokenizer, encoder, block_generator,decoder)

        model.load_from_original(pretrained)
        
        self.assertTrue(True)

import transformers

from src.attention import AttentionMessage
pretrained = transformers.GPT2LMHeadModel.from_pretrained('gpt2')
class GPT2_loading_parameters(unittest.TestCase):

    def test_loading_and_call(self):
        tokenizer = Tokenizer('gpt2')

        encoder = GPT2Encoder()
        decoder = GPT2Decoder()
        block_generator = BlockGenerator(GPT2_Block)
        model = GPT2(tokenizer, encoder, block_generator, decoder)

        model.load_from_original(pretrained)

        sample_text = "Legolas and Gimli advanced on the orcs, raising their weapons with a harrowing war cry."
        x=tokenizer(sample_text)

        graph_maker=linear_unidirectional_graph_maker(5)
        edge_index=graph_maker(x.shape[0])

        out=model.calculate_final_embedding(x,edge_index)

        self.assertEqual(out.shape,(len(x),tokenizer.vocab_size))
        self.assertFalse(out.isnan().any())
        self.assertFalse(out.isinf().any())

    def test_MLP_is_the_same(self):
        tokenizer = Tokenizer('gpt2')

        encoder = GPT2Encoder()
        decoder = GPT2Decoder()
        block_generator = BlockGenerator(GPT2_Block)
        model = GPT2(tokenizer, encoder, block_generator, decoder)

        sequence_length=17
        d_Embedding=model.d_Embedding
        x=torch.randn([sequence_length,d_Embedding])

        for i in range(model.n_blocks):
            output=model.transformer_blocks[i].MLP(x)
            target=pretrained.transformer.h[i].mlp(x)

            out=target/output
            
            where=torch.isclose(out,torch.ones_like(x)*1.111,atol=1e-2)

            out=out[where]

            self.assertTrue(torch.allclose(out,torch.ones_like(out)*1.111,atol=1e-2))

    def test_attention_GPT2_QKV(self):
        tokenizer = Tokenizer('gpt2')

        encoder = GPT2Encoder()
        decoder = GPT2Decoder()
        block_generator = BlockGenerator(GPT2_Block)
        model = GPT2(tokenizer, encoder, block_generator, decoder)

        model.load_from_original(pretrained)

        sequence_length=17
        batch_size=1
        n_heads=12
        d_Embedding=64*n_heads
        


        graph_maker=linear_unidirectional_graph_maker(40)
        edge_index=graph_maker(sequence_length)
        senders,recievers=edge_index

        

        for i in range(model.n_blocks):

            Q=torch.randn([batch_size,n_heads,sequence_length,d_Embedding//n_heads])*50
            K=torch.randn([batch_size,n_heads,sequence_length,d_Embedding//n_heads])*50
            V=torch.randn([batch_size,n_heads,sequence_length,d_Embedding//n_heads])*20
            out_pretrained,att_pretrained=pretrained.transformer.h[i].attn._attn(Q,K,V)
            att_pretrained=att_pretrained.view(n_heads,sequence_length,sequence_length)
            
            Q=Q.permute(0,2,1,3).view(sequence_length,n_heads,d_Embedding//n_heads)
            K=K.permute(0,2,1,3).view(sequence_length,n_heads,d_Embedding//n_heads)
            V=V.permute(0,2,1,3).view(sequence_length,n_heads,d_Embedding//n_heads)
            attention_message=AttentionMessage()

            out,att=attention_message(Q,K,V,edge_index)
            att=att.permute(1,0)

            for j in range(att_pretrained.shape[-1]):

                self.assertTrue(
                    torch.allclose(att_pretrained[:,j:,j],att[:,senders==j],1e-3,1e-3)
                )

    def test_attention_GPT2_out(self):
        tokenizer = Tokenizer('gpt2')

        encoder = GPT2Encoder()
        decoder = GPT2Decoder()
        block_generator = BlockGenerator(GPT2_Block)
        model = GPT2(tokenizer, encoder, block_generator, decoder)

        model.load_from_original(pretrained)

        sequence_length=17
        batch_size=1
        n_heads=12
        d_Embedding=64*n_heads
        


        graph_maker=linear_unidirectional_graph_maker(40)
        edge_index=graph_maker(sequence_length)
        senders,recievers=edge_index

        

        for i in range(model.n_blocks):

            Q=torch.randn([batch_size,n_heads,sequence_length,d_Embedding//n_heads])
            K=torch.randn([batch_size,n_heads,sequence_length,d_Embedding//n_heads])
            V=torch.randn([batch_size,n_heads,sequence_length,d_Embedding//n_heads])
            out_pretrained,att_pretrained=pretrained.transformer.h[i].attn._attn(Q,K,V)
            att_pretrained=att_pretrained.view(n_heads,sequence_length,sequence_length)
            out_pretrained=out_pretrained.permute(0,2,1,3).view(sequence_length,n_heads,d_Embedding//n_heads)

            Q=Q.permute(0,2,1,3).view(sequence_length,n_heads,d_Embedding//n_heads)
            K=K.permute(0,2,1,3).view(sequence_length,n_heads,d_Embedding//n_heads)
            V=V.permute(0,2,1,3).view(sequence_length,n_heads,d_Embedding//n_heads)
            attention_message = AttentionMessage()
            out,att=attention_message(Q,K,V,edge_index)
            att=att.permute(1,0)


            self.assertTrue(
                torch.allclose(out_pretrained,out,1e-3,1e-3)
            )


    def test_make_QKV_GPT2(self):
        tokenizer = Tokenizer('gpt2')

        encoder = GPT2Encoder()
        decoder = GPT2Decoder()
        block_generator = BlockGenerator(GPT2_Block)
        model = GPT2(tokenizer, encoder, block_generator, decoder)

        model.load_from_original(pretrained)

        

        sequence_length=17
        batch_size=1
        n_heads=12
        d_Embedding=64*n_heads

        graph_maker=linear_unidirectional_graph_maker(40)
        edge_index=graph_maker(sequence_length)
        senders,recievers=edge_index

        def split_heads(tensor, num_heads, attn_head_size):
            """
            Splits hidden_size dim into attn_head_size and num_heads
            """
            new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
            tensor = tensor.view(new_shape)
            return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        
        def merge_heads(tensor, num_heads, attn_head_size):
            """
            Merges attn_head_size dim and num_attn_heads dim into hidden_size
            """
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
            new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
            return tensor.view(new_shape)

        def original_c_attn(x,i=0):
            query, key, value = pretrained.transformer.h[i].attn.c_attn(x).split(d_Embedding, dim=2)

            dim=d_Embedding//n_heads
            query = split_heads(query, n_heads, dim)
            key = split_heads(key, n_heads, dim)
            value = split_heads(value, n_heads, dim)

            return query,key,value
       
        for i in range(model.n_blocks):
            x=torch.randn([1,sequence_length,d_Embedding])
            out_pretrained=original_c_attn(x,i)

            out_pretrained_reshaped=[
                thing.view([n_heads,sequence_length,d_Embedding//n_heads]).permute(1,0,2) 
                for thing in out_pretrained]

            x=x.view(x.shape[1:])
            out=model.transformer_blocks[i].attention_block.make_QKV(x)
            for thing, thing_pretrained in zip(out,out_pretrained_reshaped):
                self.assertTrue(torch.allclose(thing,thing_pretrained,1e-3,1e-3))

            Qp,Kp,Vp=out_pretrained
            Q, K, V =out

            out_pretrained,att_pretrained=pretrained.transformer.h[i].attn._attn(Qp,Kp,Vp)
            out_pretrained_reshaped=out_pretrained.permute(0,2,1,3).view(sequence_length,n_heads,d_Embedding//n_heads)
            attention_message = AttentionMessage()
            out,att=attention_message(Q,K,V,edge_index)
            
            self.assertTrue(torch.allclose(out,out_pretrained_reshaped,1e-3,1e-3))

            out_pretrained = merge_heads(out_pretrained, n_heads, d_Embedding//n_heads)
            out_pretrained = pretrained.transformer.h[i].attn.c_proj(out_pretrained)

            out=model.transformer_blocks[i].attention_block.feedforward(out)

            self.assertTrue(torch.allclose(out,out_pretrained,1e-3,1e-3))


    def test_attention_is_the_same(self):
        tokenizer = Tokenizer('gpt2')

        encoder = GPT2Encoder()
        decoder = GPT2Decoder()
        block_generator = BlockGenerator(GPT2_Block)
        model = GPT2(tokenizer, encoder, block_generator, decoder)

        model.load_from_original(pretrained)

        sequence_length=17
        batch_size=1
        n_heads=12
        d_Embedding=64*n_heads

        graph_maker=linear_unidirectional_graph_maker(40)
        edge_index=graph_maker(sequence_length)
        senders,recievers=edge_index

        for i in range(model.n_blocks):
            x=torch.randn([1,sequence_length,d_Embedding])
            target,_=pretrained.transformer.h[i].attn(x)
            x.view(size=(sequence_length,d_Embedding))
            output=model.transformer_blocks[i].attention_block(x,edge_index)

            self.assertTrue(torch.allclose(output,target.view(sequence_length,d_Embedding),1e-3,1e-3))


    def test_attention_block_is_equal(self):
        tokenizer = Tokenizer('gpt2')

        encoder = GPT2Encoder()
        decoder = GPT2Decoder()
        block_generator = BlockGenerator(GPT2_Block)
        model = GPT2(tokenizer, encoder, block_generator, decoder)

        model.load_from_original(pretrained)

        sequence_length=17
        batch_size=1
        n_heads=12
        d_Embedding=64*n_heads

        graph_maker=linear_unidirectional_graph_maker(40)
        edge_index=graph_maker(sequence_length)
        senders,recievers=edge_index
        for i in range(model.n_blocks):
            x=torch.randn([1,sequence_length,d_Embedding])
            target=pretrained.transformer.h[i](x)
            target=target[0].view(sequence_length,d_Embedding)
            x.view(size=(sequence_length,d_Embedding))
            output=model.transformer_blocks[i](x,edge_index)

            self.assertTrue(torch.allclose(output,target,1e-3,1e-3))

    def test_all_attention_block_are_equal(self):
        tokenizer = Tokenizer('gpt2')

        encoder = GPT2Encoder()
        decoder = GPT2Decoder()
        block_generator = BlockGenerator(GPT2_Block)
        model = GPT2(tokenizer, encoder, block_generator, decoder)

        model.load_from_original(pretrained)

        sequence_length=17
        batch_size=1
        n_heads=12
        d_Embedding=64*n_heads

        graph_maker=linear_unidirectional_graph_maker(40)
        edge_index=graph_maker(sequence_length)

        x_p=torch.randn([1,sequence_length,d_Embedding])
        x=x_p.view(size=(sequence_length,d_Embedding)).clone()
        for i in range(model.n_blocks):
            x_p=pretrained.transformer.h[i](x_p)[0]
            x=model.transformer_blocks[i](x,edge_index)

        self.assertTrue(torch.allclose(x,x_p,1e-3,1e-3))



    def test_all_GPT2(self):
        tokenizer = Tokenizer('gpt2')

        encoder = GPT2Encoder()
        decoder = GPT2Decoder()
        block_generator = BlockGenerator(GPT2_Block)
        model = GPT2(tokenizer, encoder, block_generator, decoder)
        graph_maker=linear_unidirectional_graph_maker(40)

        model.load_from_original(pretrained)

        x = tokenizer("Legolas and Gimli advanced on the orcs, raising their weapons with a harrowing war cry.")
        
        edge_index=graph_maker(x.shape[0])
        senders,recievers=edge_index

        x_p=pretrained(x)[0] #logtis
        x=model.calculate_final_embedding(x,edge_index) #logits

        self.assertTrue(torch.allclose(x,x_p,1e-3,1e-3))
    

