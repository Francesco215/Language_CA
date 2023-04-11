import torch
import numpy as np

BITS=64
#most of the code has been adapted from my old repository about Quantum-Diffusion
#https://github.com/Francesco215/Quantum-diffusion/blob/main/src/diffusion_utils.py


#This part is for the scheduling of the alphas
#TODO: check if this function is consistend with the reverse step definition
def linear_schedule(t:float ,t_max:float, bits=BITS):
    """Calculates the alpha value for a given timestep, see eq. 17 of improved DDPM paper

    Args:
        t (float): current timestep
        t_max (float): total number of timesteps

    Returns:
        float: alpha value
    """

    s=1/bits

    return (t/t_max+s)/(1+s)

#TODO: check if this function is consistend with the reverse step definition
def cosine_schedule(t:float ,t_max:float, bits=BITS):
    """Calculates the alpha value for a given timestep, see eq. 17 of improved DDPM paper

    Args:
        t (float): current timestep
        t_max (float): total number of timesteps

    Returns:
        float: alpha value
    """

    s=1/bits

    #TODO: check if using the torch implementation for the cosine function is better
    return np.cos((t/t_max+s)/(1+s)*np.pi/2)**2



#this part is for the denoising
def denoise(model, reverse_step_function, x, time, timesteps, schedule, **kwargs) -> torch.Tensor:
    """Generates an image from pure noise

    Args:
        forward (Callable): the forward function of the diffusion model diffusion.forward()
        x (torch.Tensor): text to denoise (...,n,h,c) 
        time (float): the current timestep
        timesteps (float): the number of total timesteps to count
        schedule (Callable): the function to use for the scheduling of the alphas
    Returns:
        torch.Tensor: The the denoised input
    """
    #TODO: check the last step
    alpha_old = schedule(time, timesteps)
    alpha_next=schedule(time-1,timesteps)*torch.ones(len(x)).to(x.device)

    for t in range(time-2,0,-1):
        # predict
        noise_encoding = model.encoder.noise_encoder(alpha_old)
        x_0 = model(x+noise_encoding, **kwargs)


        alpha_old=alpha_next
        alpha_next=schedule(t,timesteps)*torch.ones(len(x)).to(x.device)

        # reverse step
        x=reverse_step_function(x, x_0, alpha_old, alpha_next)
        
    noise_encoding = model.encoder.noise_encoder(alpha_old)
    x=model(x+noise_encoding, **kwargs)
    return x


# Utils for diffusion
def generate_from_noise(forward, reverse_step_function, shape, timesteps, schedule, device, self_conditioning) -> torch.Tensor:
    """Generates an image from pure noise

    Args:
        forward (Callable): the forward function of the diffusion model diffusion.forward()
        reverse_step_function (Callable): the function to use for the reverse step
        shape (float): the shape of the tensor to generate. (b,c,h,w)
            the first dimention is the batch
            the second dimention represents the channels, it must be equal to 3*BITS
            the third and fourth dimention represent the height and width of the image
        timesteps (float): the number of total timesteps to count
        schedule (Callable): the function to use for the scheduling of the alphas
        device (torch.Device): the device to use
        k (float): it is a parameter that changes the way the gaussian noise id added
        collapsing (bool): if True then the qubits are collapsed after each timestep
    Returns:
        torch.Tensor: The generated images
    """

    x=torch.randn(shape, device=device)
    return denoise(forward, reverse_step_function, x, timesteps, timesteps, schedule, self_conditioning)



#this part defines the reverse step
def reverse_step(x: torch.tensor, x_0:torch.tensor, alpha_old:float, alpha_next:float, sigma:float) -> torch.Tensor:
    """Does the reverse step, you must calculate the prediction x_0 separatelly. It implements eq 12 of the DDIM paper

    Args:
        x (torch.tensor): current state of the input
        x_0 (torch.tensor): prediction of the original input
        alpha_old (float): see eq 12 of DDIM paper
        alpha_next (float): see eq 12 of DDIM paper
        sigma (int or torch.Tensor, optional): Noise of the reverse step,
            if sigma = 0 then it is a DDIM step,
            if sigma = sqrt((1-alpha_old)/alpha_old) then it is a DDPM step.
            Defaults to 0.

    Returns:
        torch.Tensor: reverse step
    """
    mean=bmult(torch.sqrt(alpha_next),x_0)
    
    dx=x-bmult(torch.sqrt(alpha_old),x_0)
    const=torch.sqrt((1-sigma**2-alpha_next)/(1-alpha_old))
    
    mean+=bmult(const,dx)

    #      mean + normal(mean=0,std=sigma, size=x.shape)
    return mean + torch.randn(x.shape, device=x.device)*sigma
    

def reverse_step_epsilon(x: torch.tensor, epsilon:torch.tensor, alpha_old:float, alpha_next:float, sigma:float) -> torch.Tensor:
    """Does the reverse step, you must calculate the noise epsilon separately. It implements eq 12 of the DDIM paper

    Args:
        x (torch.tensor): current state
        epsilon (torch.tensor): the prediction of the noise added to x
        alpha_old (float): see eq 12 of DDIM paper
        alpha_next (float): see eq 12 of DDIM paper
        sigma (int or torch.Tensor, optional): Noise of the reverse step,
            if sigma = 0 then it is a DDIM step,
            if sigma = sqrt((1-alpha_old)/alpha_old) then it is a DDPM step.
            Defaults to 0.

    Returns:
        torch.Tensor: reverse step
    """
    #dx = -sqrt(1-alpha_old)*epsilon
    dx = bmult(torch.sqrt( 1-alpha_old ), - epsilon)
    
    # mean = (x+dx)*sqrt(alpha_old/alpha_next) + epsilon*sqrt(1 - sigma**2 - alpha_old)
    mean = bmult(torch.sqrt( alpha_old/alpha_next ),x + dx)
    mean += bmult(torch.sqrt( 1 - sigma**2 - alpha_old ) , epsilon)
    #      mean + normal(mean=0,std=sigma, size=x.shape)
    return mean + bmult(sigma, torch.normal(0, 1, size=x.shape, device=x.device))



def reverse_DDIM(x: torch.tensor, x_0:torch.tensor, alpha_old:float, alpha_next:float) -> torch.Tensor:
    """Calculates the reverse step. It implements eq 12 of the DDIM paper

    Args:
        x (torch.tensor): current state of the image
        x_0 (torch.tensor): prediction of the original image
        alpha_old (float): see eq 12 of DDIM paper
        alpha_next (float): see eq 12 of DDIM paper

    Returns:
        torch.Tensor: reverse step
    """
    return reverse_step(x,x_0,alpha_old,alpha_next,0)

def reverse_DDPM(x: torch.tensor, x_0:torch.tensor, alpha_old:float, alpha_next:float) -> torch.Tensor:
    """Calculates the reverse step. It implements eq 12 of the DDIM paper

    Args:
        x (torch.tensor): current state of the image
        x_0 (torch.tensor): prediction of the original image
        alpha_old (float): see eq 12 of DDIM paper
        alpha_next (float): see eq 12 of DDIM paper

    Returns:
        torch.Tensor: reverse step
    """
    sigma=torch.sqrt((1-alpha_old)/alpha_old)
    return reverse_step(x,x_0,alpha_old,alpha_next,sigma)



def bmult(batch_wise_vector, tensor) -> torch.Tensor:
    """Multiplies a vector for a tensor over the first dimention.
       it is used for multiplying each batch for a different number
    Args:
        batch_wise_vector (torch.Tensor or float): vector or scalar
        tensor (torch.Tensor): tensor

    Returns:
        torch.Tensor: tensor with the same dimentions of x
    """
    if type(batch_wise_vector)==float or type(batch_wise_vector)==int or len(batch_wise_vector)==1:
        return batch_wise_vector*tensor

    return torch.einsum("b , b ... -> b ...",batch_wise_vector,tensor)