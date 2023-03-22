import torch
from torch import nn
import transformers
from transformers import AutoTokenizer
from src.encoder import Encoder, GPT2Encoder
from src.decoder import Decoder, GPT2Decoder
from src.graph_initialization import linear_unidirectional_graph_maker
from src.graphAN import GraphAttentionNetwork, BlockGenerator
from src.tokenizer import Tokenizer
from src.GPT2 import GPT2_Block, GPT2

device='cpu'
ce_loss = nn.CrossEntropyLoss()


pretrained = transformers.GPT2LMHeadModel.from_pretrained('gpt2').to(device)

tokenizer=AutoTokenizer.from_pretrained('gpt2')
text = "The capital of France is Paris, and"
encoded_text=tokenizer.encode(text,add_special_tokens=False,return_tensors='pt').to(device)

for _ in range(10):
    logits_last_token=pretrained(encoded_text).logits[0][-1]
    last_token=logits_last_token.argmax(-1)
    encoded_text=torch.cat((encoded_text,last_token.view(1,1)),dim=1)

print(tokenizer.decode(encoded_text[0]))

print('\n\n')


tokenizer = Tokenizer('gpt2',device=device)
encoder = GPT2Encoder()
decoder = GPT2Decoder(encoder)
block_generator = BlockGenerator(GPT2_Block)
model = GPT2(tokenizer, encoder, block_generator, decoder)


model.load_from_original(pretrained)

graph_maker = linear_unidirectional_graph_maker(1000)

encoded_text=encoded_text[0]
edge_index=graph_maker(encoded_text.shape[0])

out=model.generate_most_prob(text,10,graph_maker)

print(out)


