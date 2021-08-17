from .utils import *
from .dust_attenuation import AttenuationCurve, DustAttenuation
from .dust_emission import DustEmission

import pickle

import numpy as np
from astropy import units as U
from astropy import constants
import torch
from torch import nn


class MultiwavelengthSED(nn.Module):
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
    def __init__(self, helper, lib_ssp, dust_attenuation, dust_emission, converter, adapter=None):
        super().__init__()
        self.helper = helper
        self.lib_ssp = lib_ssp
        self.dust_attenuation = dust_attenuation
        self.dust_emission = dust_emission
        self.converter = converter
        if adapter is None:
            self.adapter = Adapter(helper)
        else:
            self.adapter = adapter
        self.lam = converter.lam
        self.lam_pivot = converter.lam_pivot
        self.return_ph = self.lam_pivot is not None


    @classmethod
    def from_checkpoint(
        cls, helper, lib_ssp, fname_da_disk, fname_da_bulge, fname_de, converter, adapter=None
    ):
        curve_disk, _ = load_model(fname_da_disk, AttenuationCurve)
        curve_bulge, _ = load_model(fname_da_bulge, AttenuationCurve)
        dust_attenuation = DustAttenuation(helper.lookup, curve_disk, curve_bulge, lib_ssp.l_ssp)
        dust_emission = DustEmission.from_checkpoint(fname_de, L_ssp=lib_ssp.L_ssp)
        return cls(helper, lib_ssp, dust_attenuation, dust_emission, converter, adapter)

        
    def forward(self, *args):
        params, sfh_disk, sfh_bulge = self.adapter(*args)
        l_main = self.dust_attenuation(params, sfh_disk, sfh_bulge)
        l_dust_slice, frac = self.dust_emission(params, sfh_disk, sfh_bulge)
        l_dust = self.helper.set_item(torch.zeros_like(l_main), 'slice_lam_de', l_dust_slice)
        l_norm = self.helper.recover(params, 'l_norm')[:, None]
        l_tot = l_norm*(l_main + frac*l_dust)
        return self.converter(l_tot, self.return_ph)


    def predict(self, *args, return_ph=False):
        self.return_ph = return_ph
        retval = self(*args)
        self.return_ph = self.lam_pivot is not None
        return retval


    def reshape_sfh(self, sfh):
        return torch.atleast_2d(sfh).reshape((-1, *self.lib_ssp.sfh_shape))


    def sum_over_age(self, sfh):
        return self.reshape_sfh(sfh).sum(dim=self.lib_ssp.dim_age)


    def sum_over_age(self, sfh):
        return self.reshape_sfh(sfh).sum(dim=self.lib_ssp.dim_met)


class Adapter(nn.Module):
    """Apply different parametrisation to input parameters"""
    def __init__(self, helper, lib_ssp, input_mode='none', transform=None, **kwargs):
        super().__init__()
        n_sfh_fixed = 0
        inds_fixed = []
        params_fixed = []
        for key, val in kwargs.items():
            if key in helper.header:
                inds_fixed.append(helper.lookup[key])
                params_fixed.append(val)
            elif key == 'sfh_disk' or key == 'sfh_bulge':
                self.register_buffer(f"{key}_fixed", val)
                n_sfh_fixed += 1
            else:
                KeyError(f"Unknown parameter: {key}.")
        self.n_params = len(helper.header)
        self.n_sfh_fixed = n_sfh_fixed
        self.inds_fixed = tuple(inds_fixed)
        self.inds_unfixed = tuple([i_p for i_p in range(self.n_params) if i_p not in inds_fixed])
        self.register_buffer("params_fixed", torch.tensor(params_fixed, dtype=torch.float32))
        if not hasattr(self, 'sfh_disk_fixed'):
            self.sfh_disk_fixed = None
        if not hasattr(self, 'sfh_bulge_fixed'):
            self.sfh_bulge_fixed = None
        self._set_free_shapes(lib_ssp)
        self.input_mode = input_mode
        self.transform = transform


    def forward(self, *args):
        return self.set_fixed_params(self.preprocess(*args))


    def preprocess(self, *args):
        if self.input_mode == 'numpy':
            args = torch.as_tensor(args[0], dtype=torch.float32)
            free_params = self.unflatten(args)
        elif self.input_mode == 'flat':
            free_params = self.unflatten(args[0])
        elif self.input_mode == 'none':
            free_params = args
        else:
            raise ValueError("Unknown mode: {}".format(self.input_mode))
        if self.transform is not None:
            free_params = self.transform(*free_params)
            assert(type(free_params) is tuple)
        return free_params


    def set_fixed_params(self, free_params):
        gp_in = free_params[0]
        n_in = gp_in.size(0)
        gp = torch.zeros([n_in, self.n_params], dtype=gp_in.dtype, device=gp_in.device)
        gp[:, self.inds_fixed] = self.params_fixed
        gp[:, self.inds_unfixed] = gp_in
        if self.n_sfh_fixed == 2:
            sfh_disk = self.sfh_disk_fixed.tile((n_in, 1))
            sfh_bulge = self.sfh_bulge_fixed.tile((n_in, 1))
        elif self.n_sfh_fixed == 1:
            if self.sfh_disk_fixed is None:
                sfh_disk = free_params[1]
                sfh_bulge = self.sfh_bulge_fixed.tile((n_in, 1))
            if self.sfh_bulge_fixed is None:
                sfh_disk = self.sfh_disk_fixed.tile((n_in, 1))
                sfh_bulge = free_params[1]
        elif self.n_sfh_fixed == 0:
            sfh_disk = free_params[1]
            sfh_bulge = free_params[2]
        return gp, sfh_disk, sfh_bulge


    def unflatten(self, x_in):
        free_shapes = self.free_shapes 
        x_in = torch.atleast_2d(x_in)
        x_out = [None]*len(free_shapes)
        idx_b = 0
        for i_input, size in enumerate(free_shapes):
            idx_e = idx_b + size
            x_out[i_input] = x_in[:, idx_b:idx_e]
            idx_b = idx_e
        return tuple(x_out)


    def _set_free_shapes(self, lib_ssp):
        free_shapes = [len(self.inds_unfixed)]
        for i_sfh in range(2 - self.n_sfh_fixed):
            free_shapes.append(lib_ssp.l_ssp.size(0))
        self.free_shapes = tuple(free_shapes)



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
        self._set_unit(distmod)
        lam = lam*(1 + z)
        if filters is None:
            self.lam_pivot = None
        else:
            trans_filter = np.zeros([len(filters), len(lam)])
            lam_pivot = np.zeros(len(filters))
            for i_f, ftr in enumerate(filters):
                lam_pivot[i_f], trans_filter[i_f] = self._set_transmission(ftr, lam)
            self.register_buffer('trans_filter', torch.tensor(trans_filter, dtype=torch.float32))
            self.register_buffer('lam_pivot', torch.tensor(lam_pivot, dtype=torch.float32))
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


    def _set_transmission(self, ftr, lam):
        """The given flux should be in L_sol."""
        lam_pivot = ftr.wave_pivot
        trans = np.interp(lam, ftr.wavelength, ftr.transmission, left=0., right=0.)
        trans = trans/np.trapz(trans/lam, lam)*self.unit_ab
        return lam_pivot, trans


    def _set_unit(self, distmod):
        unit_f_nu = U.solLum/(4*np.pi*(10*U.parsec)**2)*U.angstrom/constants.c*10**(-.4*distmod)
        self.unit_f_nu = unit_f_nu.to(U.jansky).value
        self.unit_ab = self.unit_f_nu/3631


class SSPLibrary(nn.Module):
    def __init__(self, fname, log_lam, eps=5e-4):
        lib_ssp = pickle.load(open(fname, "rb"))
        lam_ssp = lib_ssp['lam']
        log_lam_ssp = np.log(lam_ssp)
        l_ssp_raw = lib_ssp['flx']/lib_ssp['norm']
        self.sfh_shape = l_ssp_raw.shape[1:] # (lam, met, age)
        self.dim_age = 2
        self.dim_met = 1
        l_ssp_raw.resize(l_ssp_raw.shape[0], l_ssp_raw.shape[1]*l_ssp_raw.shape[2])
        l_ssp_raw = l_ssp_raw.T
        l_ssp_raw *= lam_ssp
        L_ssp = reduction(l_ssp_raw, log_lam_ssp, eps=eps)[0]
        l_ssp = interp_arr(log_lam, log_lam_ssp, l_ssp_raw)
        # Save attributes
        super().__init__()
        self.register_buffer('tau', torch.tensor(lib_ssp['tau'], dtype=torch.float32))
        self.register_buffer('met', torch.tensor(lib_ssp['met'], dtype=torch.float32))
        self.register_buffer('log_lam', torch.tensor(log_lam, dtype=torch.float32))
        self.register_buffer('l_ssp', torch.tensor(l_ssp, dtype=torch.float32))
        self.register_buffer('L_ssp', torch.tensor(L_ssp, dtype=torch.float32))


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

