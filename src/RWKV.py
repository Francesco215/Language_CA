import torch
from torch import nn
from torch.nn import functional as F


class MixingLinear(nn.Module):
    def __init__(self, in_features,out_feature,bias=True,device='cpu',dtype=None):
        super().__init__(in_features,out_feature,bias,device,dtype)

        #TODO: check if the mixing_constant is trained
        self.mixing_constant=nn.Parameter(torch.rand(()))

    def forward(self, x1, x2):
        x = self.mixing_constant*x1 + (1-self.mixing_constant)*x2
        return super().forward(x)



#code adapted from https://johanwind.github.io/2023/03/23/rwkv_details.html
class ChannelMixing(nn.Module):
    def __init__(self,d_Embedding,dK):

        self.Kmaker=MixingLinear(d_Embedding,dK)
        self.Kactiv=nn.ReLU()

        self.Rmaker=MixingLinear(d_Embedding,dK)
        self.Ractiv=nn.Sigmoid()

        self.Vmaker=nn.Linear(dK,d_Embedding)

    def forward(self,x,x_prev):
        K=self.Kmaker(x,x_prev)
        K=self.Kactiv(K**2)

        R=self.Rmaker(x,x_prev)
        R=self.Ractiv(R)

        VK=self.Vmaker(K)

        return R*VK

#TODO: check if the weights are shareb between the MixingLinear weights of the ChannelMixing class and the TimeMixing class

class TimeMixing(nn.Module):

    def __init__(self, d_Embedding, dK):
        self.Kmaker=MixingLinear(d_Embedding,dK)

        self.Rmaker=MixingLinear(d_Embedding,dK)
        self.Ractiv=nn.Sigmoid()

        self.Vmaker=MixingLinear(d_Embedding,dK)
        
        self.final=nn.Linear(dK,d_Embedding)

        self.bonus=torch.randn(dK,requires_grad=True)
        self.decay=torch.tensor(.2,requires_grad=True)


    def forward(self,x,x_prev,num,den):

        K=self.Kmaker(x,x_prev)
        R=self.Kmaker(x,x_prev)
        V=self.Kmaker(x,x_prev)

        WKV=(num + torch.exp(self.bonus+K)*V)/(den + torch.exp(self.bonus+K)) 

        RWKV=self.Ractiv(R)*WKV

        num=torch.exp(-torch.exp(self.decay))*num + torch.exp(K)*V
        den=torch.exp(-torch.exp(self.decay))*den + torch.exp(K)

        return self.final(RWKV), num, den



class RWKV(nn.Module):

    def __init__(self,d_Embedding,dK):
        super().__init__(self,d_Embedding,dK)

        self.layernorm1=nn.LayerNorm()
        self.layernorm2=nn.LayerNorm()

        self.channelmixing=ChannelMixing(d_Embedding,dK)
        self.timemixing=TimeMixing(d_Embedding,dK)

        self.x_prev=torch.zeros(1,d_Embedding)

    def forward(self,x):
        x=self.layernorm1(x)
        x=self.channelmixing(x,self.x_prev)
        x=self.layernorm2(x)
        x, num, den =self.timemixing(x, self.x_prev, num, den)

        return x
    