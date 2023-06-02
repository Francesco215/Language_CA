import einops
from numpy.random import randint
import random
import torch
from torch.utils.data import Dataset


from src.cellular_automata import CellularAutomata

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

    def __call__(self,lenght=None,train=True):
        """Samples text and encodes it

        Args:
            train (bool, optional): By default it samples from the training set,
            if false it samples from the validation set.

        Returns:
            torch.Tensor: The encoded text
        """
        if lenght is None: lenght=self.lenght
        data = self.train_data if train else self.val_data
        
        starting_index=randint(0,len(data)-lenght)

        if starting_index+lenght >= len(data):
            return self.tokenizer(data[starting_index:])

        return self.tokenizer(data[starting_index:starting_index+lenght])


class SamplePool(Dataset):
    def __init__(self,
                 pool_size: int,
                 generator: TextGenerator,
                 model: CellularAutomata,
                 indexes_max_loss_size=32,
                 graph_maker=None) -> None:
        """Initializes the sample pool

        Args:
            pool_size (int): Number of texts in the pool
            generator (Callable): function that generates the data
            device (torch.device, optional): Device where to store the data.
                Defaults to "cpu".
            indexes_max_loss_size (int, optional): Maximum number of texts to 
                replace with freshly sampled texts. Defaults to 32.
        """
        assert generator.device == model.device, f'The device of the generator must be the same of the sample pool, instead got {generator.device}, {model.device}'
        assert isinstance(model, CellularAutomata), f"The encoder must be an instance of NoiseEncoder, got {type(model)} instead"

        self.size = pool_size
        self.generator = generator
        self.encoder=model.encoder
        self.noise_encoder=model.encoder.noise_encoder
        self.loss_function=model.loss_function
        self.device=model.device
        self.indexes_max_loss_size = indexes_max_loss_size

        self.target_tokens     = torch.empty((pool_size,generator.datapoint_shape[0]),dtype=torch.long, device=self.device)
        self.clean_embeddings  = torch.empty((pool_size,*generator.datapoint_shape),device=self.device)
        self.noised_embeddings = torch.empty((pool_size,*generator.datapoint_shape),device=self.device)
        self.noise_level       = torch.rand(pool_size,1).to(self.device)
        
        self.original_losses   = torch.zeros(pool_size, device=self.device)
        self.losses            = torch.zeros(pool_size, device=self.device)

        self.reset()

        self.all_indexes = set(range(pool_size))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: torch.Tensor) -> torch.Tensor:
        return self.target_tokens[idx], self.noised_embeddings[idx], self.clean_embeddings[idx], self.noise_encoder(self.noise_level[idx]), self.noise_level[idx]

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

        clean_texts       = einops.rearrange(clean_texts,       'b n ... -> (b n) ...')
        noised_embeddings = einops.rearrange(noised_embeddings, 'b n ... -> (b n) ...')
        clean_embeddings  = einops.rearrange(clean_embeddings,  'b n ... -> (b n) ...')
        noise_encoding    = einops.rearrange(noise_encoding,    'b e ... -> (b e) ...')

        return clean_texts, noised_embeddings, clean_embeddings, noise_encoding, idx, noise_level

    @torch.no_grad()
    def generate_data(self,indexes):
        if type(indexes)==int: indexes=[indexes]
        if indexes==None: return
        self.evolutions_per_datapoint[indexes]*=0
        for i in indexes:
            self.target_tokens[i] = self.generator()
            self.noised_embeddings[i], self.clean_embeddings[i], noise_encoding = self.encoder(self.target_tokens[i], self.noise_level[i])
            self.original_losses[i], _ = self.loss_function(self.noised_embeddings[i], self.target_tokens[i], self.clean_embeddings[i], noise_encoding)

    def reset(self):
        self.evolutions_per_datapoint = torch.zeros(self.size, dtype=torch.long,device=self.device)
        self.generate_data(range(self.size))

    @torch.no_grad()
    def update(self, indexes: List[int],
                denoised_embeddings: torch.Tensor,
                evolution_iters,
                losses):
        """Updates the data in the pool with new data at the given indexes.

        Args:
            indexes (List[int]): Indexes of the data to update
            data (torch.Tensor): New data to insert at the given indexes
            indexes_max_loss (List[int], optional): Indexes of the data with
                maximum loss, these data will be resampled.
                Default None, no data will be resampled
        """
        if indexes!= int and len(indexes!=1):
            denoised_embeddings=einops.rearrange(denoised_embeddings,'(b n) ... -> b n ...', b=len(indexes))

        self.noised_embeddings[indexes] = denoised_embeddings.to(self.device)

        self.evolutions_per_datapoint[indexes] += evolution_iters

        if type(indexes)==int:
            idx_to_generate=indexes if (losses>self.original_losses[indexes]/2).item() else None
        else: idx_to_generate=indexes[losses>self.original_losses[indexes]]

        self.losses[indexes]=losses

        self.generate_data(idx_to_generate)
