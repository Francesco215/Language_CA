import torch
from src.attention import *
from src.tests.test_utils import og_attention_message


attention_message=AttentionMessage.apply

device='cpu'
heads=1
input_size=3

d_emb=2

Q = torch.randn((input_size, heads, d_emb),device=device)
K = torch.randn((input_size, heads, d_emb), device=device)
V = torch.randn((input_size, heads, d_emb), device=device)
Q_1 = Q.clone()

Q.requires_grad=True
Q.retain_grad()



edge_index=torch.randint(0, input_size, (2, 10), device=device)

x, _ = attention_message(Q, K, V, edge_index)

x=x.mean()
x.backward()


print(Q.grad)





Q_1.requires_grad=True
Q_1.retain_grad()


x,_=og_attention_message(Q_1, K, V, edge_index)

x=x.mean()
x.backward()
print(Q.grad)
print(Q_1.grad)