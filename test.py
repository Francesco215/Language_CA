import torch
from src.attention import attention_message, overlaps
from src.tests.test_utils import og_attention_message

import einops


attention_message

device='cpu'
heads=1
input_size=3

d_emb=2

Q = torch.randn((input_size, heads, d_emb), device=device)
K = torch.randn((input_size, heads, d_emb), device=device)
V = torch.randn((input_size, heads, d_emb), device=device)
Q1 = Q.clone()

Q.requires_grad=True
Q.retain_grad()



#edge_index=torch.randint(0, input_size, (2, 10), device=device).unique(dim=1)
edge_index=torch.tensor([[0, 1, 1, 2, 2, 1],
                         [0, 0, 1, 0, 1, 2]])
senders, receivers = edge_index
#print(edge_index)
ovl = overlaps(Q, K, edge_index)

x, att = attention_message(Q, K, V, edge_index)
ovl=overlaps(Q,K,edge_index)
x=x.mean()
x.backward()



Q1.requires_grad=True
Q1.retain_grad()


x1,att1=og_attention_message(Q1, K, V, edge_index)
ovl1=einops.einsum(K, Q, 'n h d, m h d -> n m h')
x1=x1.mean()
x1.backward()
print(Q.grad)
print(Q1.grad,'\n\n')

print(att.detach().numpy())
print(att1.detach().numpy())