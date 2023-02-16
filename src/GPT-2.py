
import torch
from torch import nn

from .encoder import Encoder
from .decoder import Decoder
from .transformerMP import TransformerBlock
from .data_loader import Tokenizer

class GPT2(nn.Module):

    def __init__(self):
        super().__init__()

        pass

    def forward(self):
        pass