import numpy as np
import torch
from src.encoder import Encoder, Tokenizer
from src.GPT2 import GPT2


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

@torch.no_grad()
def validation(validation_set, model, loss_function, graph_maker, starting_index=0, n_samples=30):

    loss = torch.tensor([0.], device=model.device)
    for i in range(starting_index,starting_index+n_samples):
        text = validation_set[i]['text']
        x = model.tokenizer(text)
        if isinstance(model, GPT2):
            x = x[:1023]

        nodes = x[:-1]
        target = x[1:]

        edge_index = graph_maker(nodes.shape[0])

        out = model(nodes, edge_index)
        loss += loss_function(out, target)

    return loss/n_samples
