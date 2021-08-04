import numpy as np
from astropy import units as U
from astropy import constants
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
    converter : module
        Converter.
    """
    def __init__(self, helper, dust_attenuation, dust_emission, converter):
        super().__init__()
        self.helper = helper
        self.dust_attenuation = dust_attenuation
        self.dust_emission = dust_emission
        self.converter = converter
        self.lam = converter.lam
        self.lam_pivot = converter.lam_pivot
        self.return_ph = self.lam_pivot is not None
        
        
    def forward(self, x_in):
        l_main = self.dust_attenuation(x_in)
        l_dust_slice, frac = self.dust_emission(x_in)
        l_dust = self.helper.set_item(torch.zeros_like(l_main), 'slice_lam_de', l_dust_slice)
        l_norm = self.helper.recover(x_in[0], 'l_norm')[:, None]
        l_tot = l_norm*(l_main + frac*l_dust)
        return self.converter(l_tot, self.return_ph)


    def predict(self, x_in, return_ph=False):
        self.return_ph = return_ph
        retval = self(x_in)
        self.return_ph = self.lam_pivot is not None
        return retval


class Converter(nn.Module):
    """Apply unit conversion and filters to the input fluxes.
    
    Parameters
    ----------
    filters : array
        An array of pyphot filter instances.
    lam : tensor [AA]
        Wavelength of the input fluxes.
    """
    def __init__(self, lam, distmod=0., z=0., filters=None):
        super().__init__()
        self._set_unit()
        lam = lam*(1 + z)
        if filters is None:
            self.lam_pivot = None
        else:
            trans_filter = np.zeros([len(filters), len(lam)])
            for i_f, f in enumerate(filters):
                unit_factor = self.unit_nu_f_nu/f.AB_zero_flux
                lam_f = f.wavelength.value
                trans_f = f.transmit/np.trapz(lam_f*f.transmit, lam_f)*unit_factor
                trans_filter[i_f] = np.interp(lam, lam_f, trans_f)
            lam_pivot = torch.tensor([f.lpivot.value for f in filters], dtype=torch.float32)
            self.register_buffer('trans_filter', torch.tensor(trans_filter, dtype=torch.float32))
            self.register_buffer('lam_pivot', lam_pivot)
        self.register_buffer('lam', lam)


    def forward(self, l_target, return_ph):
        """
        Parameters
        ----------
        l_target : tensor [?]
            Generalized flux density (nu*f_nu).
        """
        if return_ph:
            return -2.5*torch.log10(torch.trapz(l_target[:, None, :]*self.trans_filter, self.lam))
        else:
            return l_target*self.lam*self.unit_f_nu


    def _set_unit(self):
        unit_nu_f_nu = U.solLum/(4*np.pi*(10*U.parsec)**2)
        unit_f_nu = unit_nu_f_nu/(constants.c/U.angstrom)
        self.unit_nu_f_nu = unit_nu_f_nu.to(U.erg/U.second/U.cm**2).value
        self.unit_f_nu = unit_f_nu.to(U.jansky).value


class Helper:
    def __init__(self, header, lookup):
        self.header = header
        self.lookup = lookup


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

