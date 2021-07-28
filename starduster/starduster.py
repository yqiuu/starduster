import torch
from torch import nn

class CompositeSED(nn.Module):
    def __init__(self, lookup, dust_attenuation, dust_emission):
        super().__init__()
        self.lookup = lookup
        self.dust_attenuation = dust_attenuation
        self.dust_emission = dust_emission
        
        
    def forward(self, x_in):
        l_main = self.dust_attenuation(x_in)
        l_dust = torch.zeros_like(l_main)
        l_dust[:, self.lookup['slice_lam_de']], frac = self.dust_emission(x_in)
        l_tot = l_main + frac*l_dust
        return l_tot

