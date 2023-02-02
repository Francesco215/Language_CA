import torch
from src import graphAN


x=torch.rand([4,8]).float()
#p_encoding=PositionalEncoding(10)
#print(x.shape,p_encoding(x).shape)
#.einsum(x,p_encoding(x),'c l, l c -> c l')

embedding_f=graphAN.InputEmbedding()

x=embedding_f(x)
