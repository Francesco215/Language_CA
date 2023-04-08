import torch
from torch import nn

from src.tokenizer import Tokenizer

class Encoder(nn.Module):
    """Layer to turn tokens into word embeddings"""

    def __init__(self,
                 d_Embedding: int = 512,
                 tokenizer=Tokenizer('gpt2'),  # defaults to the bert-base dictionary size,
                 dropout: float = 0.1,
                 device: str = 'cpu'
                 ):
        """
        This funciton takes the tokens and turns them into word embeddings.
        if the input is a string, it will be tokenized using the tokenizer.
         

        Args:
            d_embedding (int):The size of the embedding vector.
            tokenizer (Tokenizer, optional): The tokenizer to use. Defaults to Tokenizer('gpt2').
            dropout (float, optional): The dropout rate. Defaults to 0.1.
            device (str, optional): The device to use. Defaults to 'cpu'.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.d_Embedding = d_Embedding
        self.device=device

        #the embedding layer turns each token into a vector of size d_Embedding
        self.embedding = nn.Embedding(self.vocab_size, d_Embedding,device=device)
        self.dropout = nn.Dropout(dropout)

        #the number of parameters is the number of tokens times the embedding dimention
        self.n_parameters = self.vocab_size * d_Embedding

    def forward(self, x):
        #tokenize if necessary
        if type(x) == str:
            x = self.tokenizer(x)

        x = self.embedding(x)
        x = self.dropout(x)
        return x




class GPT2Encoder(Encoder):

    def __init__(self, d_Embedding=768, tokenizer=Tokenizer('gpt2'), max_position_encoding=1024, dropout=0.0, device='cpu'):
        super().__init__(d_Embedding,tokenizer,dropout,device)

        # Save the parameters
        self.max_position_encoding = max_position_encoding

        # Add positional encoding
        self.positional_encoding = nn.Embedding(max_position_encoding, d_Embedding, device=device)

        # Calculate the number of parameters
        self.n_parameters = tokenizer.vocab_size * d_Embedding + max_position_encoding * d_Embedding

    def forward(self, x):
        #tokenize if necessary
        if type(x) == str:
            x = self.tokenizer(x)

        assert x.shape[0]<self.max_position_encoding, f"The sequence is too long for the positional encoding, got {x.shape[0]} but the maximum is {self.max_position_encoding}"

        #Embedding
        indices = torch.arange(x.shape[0], device=self.device)
        x = self.embedding(x) + self.positional_encoding(indices)
        x = self.dropout(x)
        return x

    def load_from_original(self, trasformer):
        # Extract the submodules
        weight_token_embedding = trasformer.wte
        weight_positional_embedding = trasformer.wpe

        assert self.embedding.weight.shape == weight_token_embedding.weight.shape
        assert self.positional_encoding.weight.shape == weight_positional_embedding.weight.shape

        # Load the embedding layer
        self.embedding.weight = weight_token_embedding.weight
        self.positional_encoding.weight = weight_positional_embedding.weight



class NoiseEncoder(Encoder):
    def __init__(self,
                 d_Embedding: int = 512,
                 # defaults to the bert-base dictionary size,
                 tokenizer=Tokenizer('gpt2'),
                 noise_encoder:nn.Module=None,
                 dropout: float = 0.1,
                 device: str = 'cpu'
                 ):
        super().__init__(d_Embedding, tokenizer, dropout, device)

        self.noise_encoder=noise_encoder
        if noise_encoder is None:
            self.noise_encoder=nn.Sequential(
                nn.Linear(1,d_Embedding//4),
                nn.ReLU(),
                nn.Linear(d_Embedding//4,d_Embedding),
            )
        
    def forward(self, x, noise=torch.rand(())):

        if type(noise) == float:
            noise = torch.tensor(noise, device=self.device)
        
        assert noise.dim()==0, f"noise should be a scalar tensor, got a {noise.dim()}-dimensional tensor"
        assert 0<=noise.item()<=1, f"noise should be between 0 and 1, got {noise}"
        noise=noise.view(1)

        noise_encoding = self.noise_encoder(noise)
        clean_encoding = super().forward(x) 
        noised_encoding = clean_encoding*torch.sqrt(1-noise) + torch.randn_like(clean_encoding)*torch.sqrt(noise) + noise_encoding
        clean_encoding = clean_encoding + noise_encoding

        return noised_encoding, clean_encoding, noise_encoding