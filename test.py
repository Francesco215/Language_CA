import torch
from src import graphAN, graph_initialization


converter=graph_initialization.text_to_graph()
batches=converter(["Hello world! good i saw you", "How are you?"])

data=batches.get_example(0)
x=data['x']

embedding_f=graphAN.InputEmbedding()

x=embedding_f(x)
