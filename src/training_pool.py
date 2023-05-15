from numpy.random import randint
import random
import torch
from torch.utils.data import Dataset

import numpy as np
from .encoder import NoiseEncoder

from typing import Any, Iterable, List, Tuple, Callable


class ShakespeareGenerator:

    def __init__(self, input_file_path, lenght, encoder):
        self.input_file_path=input_file_path
        self.lenght=int(lenght)

        assert isinstance(encoder,NoiseEncoder), f"The encoder must be of type NoiseEncoder, got {type(encoder)} instead"
        self.encoder=encoder

        with open(input_file_path, 'r') as f:
            data = f.read()

        #this splits the dataset into train and validation
        n = len(data)
        self.train_data = data[:int(n*0.9)]
        self.val_data = data[int(n*0.9):]

    def __call__(self):
        target=self.sample_text()

        noise = torch.rand(())

        return self.encoder(target, noise)
    
    def sample_text(self,train=True):
        data = self.train_data if train else self.val_data
        
        starting_index=randint(0,len(data)-self.lenght)

        if starting_index+self.lenght >= len(data):
            return data[starting_index:]

        return data[starting_index:starting_index+self.lenght]

