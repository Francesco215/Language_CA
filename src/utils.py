import numpy as np
from torch import nn

from torch.nn import functional as F


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


class OneHot(nn.Module):
    def __init__(self, d_Embedding):
        super().__init__()
        self.d_Embedding = d_Embedding

    def forward(self, x):
        return F.one_hot(x, self.d_Embedding).float()*5
    


#String manipulations for printing output

def highlight_noising(original, noised):
    """
    Returns the second string with the different characters colored in red.
    """
    assert len(original)==len(noised), f'the two strings must have the same lenght'


    highlighted_chars = []
    for s1,s2 in zip(original,noised):
        if s1 != s2:
            highlighted_chars.append('\033[31m' + s2 + '\033[0m')
        else:
            highlighted_chars.append(s2)

    return ''.join(highlighted_chars)


def highlight_denoising(original,noised,denoised):
    """Returns a string highlighting the differences between the original, noised, and denoised strings.

    Args:
        original : the original string
        noised   : the noised string
        denoised : the denoised string

    Returns:
        A string with the same length as the input strings, where each character is colored based on its
        difference with the corresponding character in the original and denoised strings. Green is used for
        characters that are correctly denoised, red is used for characters that are incorrectly denoised,
        magenta is used for characters that are incorrectly noised, and yellow is used for characters that
        are incorrectly both noised and denoised.
    """

    assert len(original)==len(noised)==len(denoised), f'the three strings must have the same lenght'

    highlighted_chars = []
    for s1,s2,s3 in zip(original,noised,denoised):
        if   s1 == s3 != s2:
            highlighted_chars.append('\033[32m' + s3 + '\033[0m')
        elif s1 != s2 == s3:
            highlighted_chars.append('\033[31m' + s3 + '\033[0m')
        elif s1 == s2 != s3:
            highlighted_chars.append('\033[35m' + s3 + '\033[0m')
        elif s1 != s2 != s3:
            highlighted_chars.append('\033[33m' + s3 + '\033[0m')
        else:
            highlighted_chars.append(s3)

    return ''.join(highlighted_chars)


def highlight_outputs(original,noised,denoised):

    return highlight_noising(original,noised), highlight_denoising(original,noised,denoised)