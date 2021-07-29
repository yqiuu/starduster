import numpy as np
from astropy import units as U
import torch
from torch import nn


class CompositeSED(nn.Module):
    """Primary module to compute multiwavelength SEDs

    Parameters
    ----------
    helper
    dust_attenuation : module
        Dust attenuation module.
    dust_emission : module
        Dust emission module.
    lam : tensor [AA]
        Wavelength of the resulting SEDs.
    z : float
        Redshift.
    filters : array
        An array of pyphot filter instances.
    """
    def __init__(self, helper, dust_attenuation, dust_emission, lam, z=0., filters=None):
        super().__init__()
        self.helper = helper
        self.dust_attenuation = dust_attenuation
        self.dust_emission = dust_emission
        #
        lam = (1. + z)*lam
        if filters is None:
            self.filter_set = None
            self.register_buffer('lam', lam)
        else:
            self.filter_set = FilterSet(filters, lam)
            self.lam = torch.tensor([f.lpivot.value for f in filters], dtype=torch.float32)
        self.z = z
        
        
    def forward(self, x_in):
        l_main = self.dust_attenuation(x_in)
        l_dust_slice, frac = self.dust_emission(x_in)
        l_dust = self.helper.set_item(torch.zeros_like(l_main), 'slice_lam_de', l_dust_slice)
        l_tot = l_main + frac*l_dust
        if self.filter_set is None:
            return l_tot
        else:
            return self.filter_set(l_tot)


class FilterSet(nn.Module):
    """Apply filters to the input fluxes.
    
    Parameters
    ----------
    filters : array
        An array of pyphot filter instances.
    lam : tensor [AA]
        Wavelength of the input fluxes.
    """
    def __init__(self, filters, lam):
        super().__init__()
        trans_filter = np.zeros([len(filters), len(lam)])
        for i_f, f in enumerate(filters):
            lam_f = f.wavelength.value
            trans_f = f.transmit/np.trapz(lam_f*f.transmit, lam_f)
            trans_filter[i_f] = np.interp(lam, lam_f, trans_f)
        self.register_buffer('trans_filter', torch.tensor(trans_filter, dtype=torch.float32))
        self.register_buffer('lam', lam)
        
        
    def forward(self, l_target):
        """
        Parameters
        ----------
        l_target : tensor [?]
            Generalized flux density (nu*f_nu).
        """
        return torch.trapz(l_target[:, None, :]*self.trans_filter, self.lam)


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

