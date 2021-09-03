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
            self.adapter = Adapter(helper, lib_ssp)
        else:
            self.adapter = adapter
        self.lam = converter.lam
        self.lam_pivot = converter.lam_pivot
        self.return_ph = self.lam_pivot is not None


    @classmethod
    def from_checkpoint(
        cls, lib_ssp, fname_da_disk, fname_da_bulge, fname_de,
        converter, adapter=None, map_location=None
    ):
        curve_disk, _ = load_model(fname_da_disk, AttenuationCurve, map_location=map_location)
        curve_bulge, _ = load_model(fname_da_bulge, AttenuationCurve, map_location=map_location)
        dust_emission = \
            DustEmission.from_checkpoint(fname_de, lib_ssp.L_ssp, map_location=map_location)
        helper = dust_emission.helper
        dust_attenuation = DustAttenuation(helper, curve_disk, curve_bulge, lib_ssp.l_ssp)
        return cls(helper, lib_ssp, dust_attenuation, dust_emission, converter, adapter)

        
    def forward(self, *args):
        params, sfh_disk, sfh_bulge = self.adapter(*args)
        l_main = self.dust_attenuation(params, sfh_disk, sfh_bulge)
        l_dust_slice, frac = self.dust_emission(params, sfh_disk, sfh_bulge)
        l_dust = self.helper.set_item(torch.zeros_like(l_main), 'slice_lam_de', l_dust_slice)
        l_norm = self.helper.get_recover(params, 'l_norm', torch)[:, None]
        l_tot = l_norm*(l_main + frac*l_dust)
        return torch.squeeze(self.converter(l_tot, self.return_ph))


    def predict(self, *args, return_ph=False):
        self.return_ph = return_ph
        retval = self(*args)
        self.return_ph = self.lam_pivot is not None
        return retval


class Adapter(nn.Module):
    """Apply different parametrisation to input parameters"""
    def __init__(self,
        helper, lib_ssp, input_mode='none', transform=None,
        sfr_bins=None, met_type='uni_idw', **kwargs
    ):
        super().__init__()
        n_free_sfh = 2
        fixed_inds = []
        fixed_params = []
        for key, val in kwargs.items():
            if key in helper.header:
                fixed_inds.append(helper.lookup[key])
                fixed_params.append(val)
            elif key == 'sfh_disk' or key == 'sfh_bulge':
                self.register_buffer(f"{key}_fixed", val)
                n_free_sfh -= 1
            else:
                KeyError(f"Unknown parameter: {key}.")
        self.n_params = len(helper.header)
        self.n_free_sfh = n_free_sfh
        self.fixed_inds = tuple(fixed_inds)
        self.free_inds = tuple([i_p for i_p in range(self.n_params) if i_p not in fixed_inds])
        self.register_buffer("fixed_params", torch.tensor(fixed_params, dtype=torch.float32))
        if not hasattr(self, 'sfh_disk_fixed'):
            self.sfh_disk_fixed = None
        if not hasattr(self, 'sfh_bulge_fixed'):
            self.sfh_bulge_fixed = None
        self.input_mode = input_mode
        self.transform = transform
        self._set_log_met(lib_ssp)
        self._set_free_shape(lib_ssp, sfr_bins, met_type)


    def forward(self, *args):
        n_in = self.n_sfh_input
        free_params = self.preprocess(*args)
        sfh_params = [self.derive_sfh(*free_params[idx : idx+n_in]) \
            for idx in range(1, len(free_params), n_in)]
        free_params = (free_params[0], *sfh_params)
        return self.set_fixed_params(free_params)


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
        gp[:, self.fixed_inds] = self.fixed_params
        gp[:, self.free_inds] = gp_in
        if self.n_free_sfh == 0:
            sfh_disk = self.sfh_disk_fixed.tile((n_in, 1))
            sfh_bulge = self.sfh_bulge_fixed.tile((n_in, 1))
        elif self.n_free_sfh == 1:
            if self.sfh_disk_fixed is None:
                sfh_disk = free_params[1]
                sfh_bulge = self.sfh_bulge_fixed.tile((n_in, 1))
            if self.sfh_bulge_fixed is None:
                sfh_disk = self.sfh_disk_fixed.tile((n_in, 1))
                sfh_bulge = free_params[1]
        elif self.n_free_sfh == 2:
            sfh_disk = free_params[1]
            sfh_bulge = free_params[2]
        return gp, sfh_disk, sfh_bulge


    def unflatten(self, x_in):
        free_shape = self.free_shape
        x_in = torch.atleast_2d(x_in)
        x_out = [None]*len(free_shape)
        idx_b = 0
        for i_input, size in enumerate(free_shape):
            idx_e = idx_b + size
            x_out[i_input] = x_in[:, idx_b:idx_e]
            idx_b = idx_e
        return tuple(x_out)

    
    def derive_sfh(self, sfr, log_met=None):
        if self.sfr_bins is None:
            if self.met_type == 'discrete':
                sfh = sfr
            elif self.met_type == 'idw' or self.met_type == 'uni_idw':
                sfh = sfr[:, None, :]*self.derive_idw_met(log_met)
                sfh = torch.flatten(sfh, start_dim=1)
        else:
            sfh = self.derive_sfr(sfr)[:, None, :]*self.derive_idw_met(log_met)
            sfh = torch.flatten(sfh, start_dim=1)
        return sfh


    def derive_sfr(self, sfr):
        sfr_out = torch.zeros([sfr.size(0), self.n_tau],
            dtype=sfr.dtype, layout=sfr.layout, device=sfr.device)
        for i_b, (idx_b, idx_e) in enumerate(self.sfr_bins):
            sfr_out[:, idx_b:idx_e] = sfr[:, i_b, None]/(idx_e - idx_b)
        return sfr_out


    def derive_idw_met(self, log_met):
        eps = 1e-6
        inv = 1./((log_met[:, None, :] - self.log_met)**2 + eps)
        weights = inv/inv.sum(dim=1)[:, None, :]
        return weights


    def _set_free_shape(self, lib_ssp, sfr_bins, met_type):
        if sfr_bins is None and met_type == 'discrete':
            sfh_shape = [lib_ssp.n_ssp,]
        elif sfr_bins is None and met_type == 'idw':
            sfh_shape = [lib_ssp.n_tau, lib_ssp.n_met]
        elif sfr_bins is None and met_type == 'uni_idw':
            sfh_shape = [lib_ssp.n_tau, 1]
        elif sfr_bins is not None and met_type == 'idw':
            sfh_shape = [len(sfr_bins), lib_ssp.n_met]
        elif sfr_bins is not None and met_type == 'uni_idw':
            sfh_shape = [len(sfr_bins), 1]
        else:
            raise ValueError("Unknown SFH type.")
        n_sfh_input = 1 if met_type == 'discrete' else 2
        # Set attributes
        self.sfr_bins = sfr_bins
        self.met_type = met_type
        self.n_sfh_input = n_sfh_input
        self.n_tau = lib_ssp.n_tau
        self.sfh_shape = tuple(sfh_shape)
        self.free_shape = tuple([len(self.free_inds)] + sfh_shape*self.n_free_sfh)


    def _set_log_met(self, lib_ssp):
        met_sol = 0.019
        self.log_met = torch.log10(lib_ssp.met/met_sol)[:, None]


class Converter(nn.Module):
    """Apply unit conversion and filters to the input fluxes.
    
    Parameters
    ----------
    filters : array
        An array of pyphot filter instances.
    lam : tensor [micormeter]
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
        unit_lam = U.angstrom.to(U.micrometer)
        lam_pivot = ftr.wave_pivot
        trans = np.interp(lam, ftr.wavelength*unit_lam, ftr.transmission, left=0., right=0.)
        trans = trans/np.trapz(trans/lam, lam)*self.unit_ab
        return lam_pivot, trans


    def _set_unit(self, distmod):
        unit_f_nu = U.solLum/(4*np.pi*(10*U.parsec)**2)*U.micrometer/constants.c*10**(-.4*distmod)
        self.unit_f_nu = unit_f_nu.to(U.jansky).value
        self.unit_ab = self.unit_f_nu/3631


class SSPLibrary(nn.Module):
    def __init__(self, fname, lam, eps=5e-4):
        lib_ssp = pickle.load(open(fname, "rb"))
        lam_ssp = lib_ssp['lam']
        log_lam_ssp = np.log(lam_ssp)
        l_ssp_raw = lib_ssp['flx']/lib_ssp['norm']
        self.sfh_shape = l_ssp_raw.shape[1:] # (lam, met, age)
        self.n_met = len(lib_ssp['met'])
        self.n_tau = len(lib_ssp['tau'])
        self.n_ssp = len(lib_ssp['met']*lib_ssp['tau'])
        self.dim_age = 2
        self.dim_met = 1
        l_ssp_raw.resize(l_ssp_raw.shape[0], l_ssp_raw.shape[1]*l_ssp_raw.shape[2])
        l_ssp_raw = l_ssp_raw.T
        l_ssp_raw *= lam_ssp
        L_ssp = reduction(l_ssp_raw, log_lam_ssp, eps=eps)[0]
        l_ssp = interp_arr(np.log(lam), log_lam_ssp, l_ssp_raw)
        # Save attributes
        super().__init__()
        self.register_buffer('tau', torch.tensor(lib_ssp['tau'], dtype=torch.float32))
        self.register_buffer('met', torch.tensor(lib_ssp['met'], dtype=torch.float32))
        self.register_buffer('lam', torch.tensor(lam, dtype=torch.float32))
        self.register_buffer('l_ssp', torch.tensor(l_ssp, dtype=torch.float32))
        self.register_buffer('L_ssp', torch.tensor(L_ssp, dtype=torch.float32))


    def reshape_sfh(self, sfh):
        return torch.atleast_2d(sfh).reshape((-1, *self.lib_ssp.sfh_shape))


    def sum_over_age(self, sfh):
        return self.reshape_sfh(sfh).sum(dim=self.lib_ssp.dim_age)


    def sum_over_met(self, sfh):
        return self.reshape_sfh(sfh).sum(dim=self.lib_ssp.dim_met)

