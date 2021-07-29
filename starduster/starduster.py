import numpy as np
from astropy import units as U
import torch
from torch import nn


class CompositeSED(nn.Module):
    def __init__(self, helper, dust_attenuation, dust_emission):
        super().__init__()
        self.helper = helper
        self.dust_attenuation = dust_attenuation
        self.dust_emission = dust_emission
        
        
    def forward(self, x_in):
        l_main = self.dust_attenuation(x_in)
        l_dust_slice, frac = self.dust_emission(x_in)
        l_dust = self.helper.set_item(torch.zeros_like(l_main), 'slice_lam_de', l_dust_slice)
        l_tot = l_main + frac*l_dust
        return l_tot


class Helper:
    def __init__(self, header, lookup):
        self.header = header
        self.lookup = lookup
        self.unit_l_norm = U.solLum.to(U.erg/U.second)/(4*np.pi*(10*U.parsec.to(U.cm))**2)


    def get_item(self, target, key):
        return target[:, self.lookup[key]]


    def set_item(self, target, key, value):
        target[:, self.lookup[key]] = value
        return target


    def transform(self, target, key):
        target = self.get_item(target, key)
        if key == 'theta':
            target = torch.cos(np.pi/180.*target)
        elif key == 'b_o_t':
            pass
        else:
            log_min, log_max = self.header[key]
            target = (torch.log10(target) - log_min)/(log_max - log_min)
        target = 2*target - 1 # Convert from [0, 1] to [-1, 1]
        return target


    def recover(self, target, key):
        target = self.get_item(target, key)
        target = .5*(target + 1) # Convert from [-1, 1] to [0, 1]
        if key == 'theta':
            target = 180/np.pi*np.arccos(target)
        elif key == 'b_o_t':
            pass
        else:
            log_min, log_max = self.header[key]
            target = 10**(target*(log_max - log_min) + log_min)
        return target

