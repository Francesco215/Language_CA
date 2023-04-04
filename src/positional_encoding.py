import torch
import einops
from torch import nn

class RotaryEncoding(nn.Module):
    
    def __init__(self, base=1e-5, thetas=None):
        super().__init__()
        self.base = base
        self.thetas = thetas
        self.cached_shape=(0,0,0,0)
        self.cached_rotary_encoding=None
        self.gamma=None



    def forward(self, x):
        """Applies a rotary embedding to a tensor.

        Args:
            x (torch.Tensor): Tensor to apply the rotary embedding to.
                x.shape=(sequence_lenght, n_heads, d_embedding)
            base (float, optional): Base of the logarithm. Defaults to 1e-5.
            thetas (torch.Tensor, optional): Tensor containing the thetas.
                It can be used in case you want to apply learned positional encoding.
                Defaults to None.

        Returns:
            torch.Tensor: Tensor with the rotary embedding applied.
        """
        assert x.shape[-1] % 2 == 0, 'the last dimension must be even'

        #pair up consecutive elements
        x1 = einops.rearrange(x, '... (n1 n2) -> ... n1 n2', n2=2)

        #pair up elements and swap them
        x2 = x1[..., torch.tensor([1, 0])]
        x2[..., 0] = -x2[..., 0]

        #create phases
        sin, cos = self.make_sin_cos(x1.shape, device=x.device)

        #apply rotation
        x1 = einops.einsum(x1, cos, 'n ... c p, n c -> n ... c p')
        x2 = einops.einsum(x2, sin, 'n ... c p, n c -> n ... c p')

        """Probably one could use this to make it work even if the input has a batch dimentions, but I haven't tested it.
        x1 = einops.einsum(x1, cos, '... h c p, ... c -> ... h c p')
        x2 = einops.einsum(x2, sin, '... h c p, ... c -> ... h c p')
        """

        x = x1+x2
        x = einops.rearrange(x, '... n1 n2 -> ... (n1 n2)', n2=2)

        return x


    def make_sin_cos(self, shape, device='cpu'):
        """
        Creates the sin and cos tensors for the rotary encoding.

        Args:
            shape (tuple): Shape of the tensor to create.
            device (str, optional): The device to use. Defaults to 'cpu'.
        """

        sequence_lenght, n_heads, d_embedding, extra_index = shape
        assert extra_index==2, 'this shoud be equal to 2, but is {}'.format(extra_index)

        if sequence_lenght <= self.cached_shape[0] and shape[1:] == self.cached_shape[1:]:
            return self.cached_rotary_encoding[:,:sequence_lenght]

        if self.thetas is None:
            self.thetas = torch.logspace(0, 1, d_embedding, self.base, device=device, requires_grad=False)
        indices = torch.arange(0, sequence_lenght, device=device)
        phases = einops.einsum(indices, self.thetas, 'a, c -> a c')

        #rotate
        sin = torch.sin(phases)
        cos = torch.cos(phases)


        if self.gamma is not None:
            zetas = torch.linspace(self.gamma,1+self.gamma,d_embedding,device=device)/(1+self.gamma)

            for i in range(sequence_lenght):
                sin[i] *= zetas**i
                cos[i] *= zetas**i

        self.cached_rotary_encoding = (sin, cos)
        return (sin, cos)


    def find_best_gamma(self,shape,device='cpu'):
        """Finds the best gamma for the rotary encoding.

        Args:
            shape (tuple): Shape of the tensor to create.
            device (str, optional): The device to use. Defaults to 'cpu'.

        Returns:
            float: The best gamma.
        """

        _,cos=self.make_sin_cos(shape,device).sum(dim=-1)

        self.gamma=0
        best_resolution=0

        for gamma in torch.linspace(0,1,100,device=device):
            resolution_=resolution(cos)
            if resolution_>best_resolution:
                best_resolution=resolution_
                self.gamma=gamma


def resolution(s:torch.Tensor):
    """Returns the resolution of a tensor.

    Args:
        s (torch.Tensor): Tensor to calculate the resolution of.

    Returns:
        int: The resolution of the tensor.
    """
    assert s.dim() == 1, 'the tensor must be 1D'

    es=torch.exp(s)
    esp,esn=es[:-1],es[1:]

    return torch.sum(esp*(esp-esn))/torch.sum(es**2)

