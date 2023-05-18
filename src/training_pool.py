from numpy.random import randint
import random
import torch
from torch.utils.data import Dataset

import numpy as np
from .encoder import NoiseEncoder

from typing import Any, Iterable, List, Tuple, Callable


class TextGenerator:

    def __init__(self, input_file_path, lenght, encoder):
        """
        Starting form the input file path, it instatiates a generator that
        samples data from the text file
        
        90% of the text is by default used for the training set, and the other 10% for the validation

        Args:
            input_file_path (str): The file path
            lenght (int): the lenght of the text that is sampled each time
            encoder (NoiseEncoder): The encoder
        """
        self.input_file_path=input_file_path
        self.lenght=int(lenght)
        self.encoder=encoder
        self.device=encoder.device
        self.tokenizer=encoder.tokenizer

        self.datapoint_shape=(self.lenght, encoder.d_Embedding)

        with open(input_file_path, 'r') as f:
            data = f.read()

        #this splits the dataset into train and validation
        n = len(data)
        self.train_data = data[:int(n*0.9)]
        self.val_data = data[int(n*0.9):]

    def __call__(self,train=True):
        """Samples text and encodes it

        Args:
            train (bool, optional): By default it samples from the training set,
            if false it samples from the validation set.

        Returns:
            torch.Tensor: The encoded text
        """
        data = self.train_data if train else self.val_data
        
        starting_index=randint(0,len(data)-self.lenght)

        if starting_index+self.lenght >= len(data):
            return self.tokenizer(data[starting_index:])

        return self.tokenizer(data[starting_index:starting_index+self.lenght])


class SamplePool(Dataset):
    def __init__(self,
                 pool_size: int,
                 generator: TextGenerator,
                 encoder: NoiseEncoder,
                 device: torch.device = "cpu",
                 indexes_max_loss_size=32) -> None:
        """Initializes the sample pool

        Args:
            pool_size (int): Number of texts in the pool
            generator (Callable): function that generates the data
            device (torch.device, optional): Device where to store the data.
                Defaults to "cpu".
            indexes_max_loss_size (int, optional): Maximum number of texts to 
                replace with freshly sampled texts. Defaults to 32.
        """
        assert generator.device==device, f'The device of the generator must be the same of the sample pool, instead got {generator.device}, {device}'
        assert isinstance(
            encoder, NoiseEncoder), f"The encoder must be an instance of NoiseEncoder, got {type(encoder)} instead"

        self.size = pool_size
        self.generator = generator
        self.encoder=encoder
        self.device=encoder.device
        self.device = device
        self.indexes_max_loss_size = indexes_max_loss_size

        self.clean_texts       = torch.empty((pool_size,generator.datapoint_shape[0]),dtype=torch.long)
        self.clean_embeddings  = torch.empty((pool_size,*generator.datapoint_shape))
        self.noised_embeddings = torch.empty((pool_size,*generator.datapoint_shape))
        self.noise_level       = torch.rand(pool_size)
        self.noise_encoding    = torch.empty(pool_size,generator.datapoint_shape[1])

        self.reset()

        self.all_indexes = set(range(pool_size))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: torch.Tensor) -> torch.Tensor:
        return self.clean_texts[idx], self.noised_embeddings[idx], self.clean_embeddings[idx], self.noise_encoding[idx], self.noise_level[idx]

    def sample(self, batch_size: int=1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples from the pool batch_size texts and returns them,
        along with the corresponding indexes

        Args:
            batch_size (int): Number of texts to extract, defaults to 1

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                The extraxted texts,
                the corresponding indexes in the sample pool
        """
        idx = torch.tensor(random.sample(self.all_indexes, batch_size))
        
        clean_texts, noised_embeddings, clean_embeddings, noise_encoding, noise_level=self.__getitem__(idx) 

        clean_texts = torch.cat(clean_texts, dim=0)
        noised_embeddings = torch.cat(noised_embeddings, dim=0)
        clean_embeddings = torch.cat(clean_embeddings, dim=0)
        noise_encoding = torch.cat(noise_encoding, dim=0)

        return clean_texts, noised_embeddings, clean_embeddings, noise_encoding, idx, noise_level

    
    #TODO vedi se serve oppure posso usare direttamente usare generate_data
    def replace_idx(self, indexes: List[int], idx_to_replace: List[int]):
        if idx_to_replace is not None:
            idx_to_replace = [indexes[i] for i in idx_to_replace]
            self.generate_data(idx_to_replace)

    def generate_data(self,indexes):
        self.evolutions_per_datapoint[indexes]*=0
        for i in indexes:
            self.clean_texts[i] = self.generator()
            self.noised_embeddings[i], self.clean_embeddings[i], self.noise_encoding[i] = self.encoder(self.clean_texts[i], self.noise_level[i])

    def reset(self):
        self.evolutions_per_datapoint = torch.zeros(self.size, dtype=torch.long)
        self.generate_data(range(self.size))

    def update(self, indexes: List[int],
                denoised_embeddings: torch.Tensor,
                idx_to_replace: List[int] = None,
                evolution_iters=None) -> None:
        """Updates the data in the pool with new data at the given indexes.

        Args:
            indexes (List[int]): Indexes of the data to update
            data (torch.Tensor): New data to insert at the given indexes
            indexes_max_loss (List[int], optional): Indexes of the data with
                maximum loss, these data will be resampled.
                Default None, no data will be resampled
        """
        self.noised_embeddings[indexes] = denoised_embeddings.detach().to(self.device)

        if evolution_iters is not None:
            self.evolutions_per_datapoint[indexes] += evolution_iters

        self.replace_idx(indexes, idx_to_replace)