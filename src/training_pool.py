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

        assert isinstance(encoder, NoiseEncoder), f"The encoder must be an instance of NoiseEncoder, got {type(encoder)} instead"
        self.encoder=encoder

        self.datapoint_shape=(self.lenght, encoder.d_Embedding)

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



class SamplePool(Dataset):
    def __init__(self,
                 pool_size: int,
                 generator: Callable[[int], torch.Tensor],
                 transform: Callable[[torch.Tensor], torch.Tensor] = lambda x:x ,
                 device: torch.device = "cpu",
                 indexes_max_loss_size=32) -> None:
        """Initializes the sample pool

        Args:
            pool_size (int): Number of texts in the pool
            generator (Callable): function that generates the data
            transform (Callable, optional): Transforms the data in some way.
                Defaults to the identity function.
            device (torch.device, optional): Device where to store the data.
                Defaults to "cpu".
            indexes_max_loss_size (int, optional): Maximum number of texts to 
                replace with freshly sampled texts. Defaults to 32.
        """
        assert generator.device==device, f'The device of the generator must be the same of the sample pool, instead got {generator.device}, {device}'

        self.size = pool_size
        self.generator = generator
        self.transform = transform
        self.device = device
        self.indexes_max_loss_size = indexes_max_loss_size

        self.data=torch.empty(pool_size,*generator.datapoint_shape)      
        for i in range(pool_size):
            self.data[i] = generator()

        self.all_indexes = set(range(pool_size))
        self.evolutions_per_datapoint = np.zeros(pool_size, dtype=int)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: torch.Tensor) -> torch.Tensor:
        return self.transform(self.data[idx])

    def transform_pool(self, transform: Callable[[torch.Tensor], torch.Tensor]):
        self.data = transform(self.data)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples from the pool batch_size texts and returns them,
        along with the corresponding indexes

        Args:
            batch_size (int): Number of texts to extract

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                The extraxted texts,
                the corresponding indexes in the sample pool
        """
        idx = random.sample(self.all_indexes - self.indexes_max_loss, batch_size) 
        
        #TODO: maybe concatenate?
        return self.transform(self.data[idx]).clone(), idx 

    def update_evolution_iters(self, indexes: List[int], evolution_iters):
        if evolution_iters is not None:
            self.evolutions_per_datapoint[indexes] += evolution_iters

    def replace_idx(self, indexes: List[int], idx_to_replace: List[int]):
        if idx_to_replace is not None:
            idx_to_replace = [indexes[i] for i in idx_to_replace]
            self.generate_data(idx_to_replace)

    def generate_data(self,indexes):
        self.evolutions_per_datapoint[indexes]*=0
        for i in indexes:
            self.data[i]=self.generator()

    def reset(self):
        self.evolutions_per_datapoint *= 0
        for i in range(self.size):
            self.data[i] = self.generator()

    def update(self, indexes: List[int],
                data: torch.Tensor,
                idx_to_replace: List[int] = None,
                evolution_iters=None) -> None:
        """Updates the data in the pool with new data at the given indexes.

        Args:
            indexes (List[int]): Indexes of the data to update
            data (torch.Tensor): New data to insert at the given indexes
            indexes_max_loss (List[int], optional): Indexes of the data with
                maximum loss, these data will be replaced with seed states.
                Default None, no data will be resampled
        """
        self.data[indexes] = data.detach().to(self.device)

        self.update_evolution_iters(indexes, evolution_iters)

        self.replace_idx(indexes, idx_to_replace)