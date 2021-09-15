from .utils import *
from .selector import Selector
from .dust_attenuation import AttenuationCurve, DustAttenuation
from .dust_emission import DustEmission

import pickle

import numpy as np
from astropy import units as U
from astropy import constants
import torch
from torch import nn
from torch.nn import functional as F


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
    detector : module
        Detector.
    """
    def __init__(self,
        helper, lib_ssp, dust_attenuation, dust_emission,
        selector_disk=None, selector_bulge=None,
    ):
        super().__init__()
        self.helper = helper
        self.lib_ssp = lib_ssp
        self.dust_attenuation = dust_attenuation
        self.dust_emission = dust_emission
        self.adapter = Adapter(helper, lib_ssp)
        self.detector = Detector(lib_ssp.lam)
        if selector_disk is not None:
            self.selector_disk = selector_disk
        if selector_bulge is not None:
            self.selector_bulge = selector_bulge
        self.return_ph = True


    @classmethod
    def from_checkpoint(
        cls, lib_ssp, fname_da_disk, fname_da_bulge, fname_de,
        fname_selector_disk, fname_selector_bulge, map_location=None
    ):
        curve_disk, _ = load_model(fname_da_disk, AttenuationCurve, map_location=map_location)
        curve_bulge, _ = load_model(fname_da_bulge, AttenuationCurve, map_location=map_location)
        selector_disk, _ = load_model(fname_selector_disk, Selector, map_location=map_location)
        selector_bulge, _ = load_model(fname_selector_bulge, Selector, map_location=map_location)
        dust_emission = \
            DustEmission.from_checkpoint(fname_de, lib_ssp.L_ssp, map_location=map_location)
        helper = dust_emission.helper
        dust_attenuation = DustAttenuation(helper, curve_disk, curve_bulge, lib_ssp.l_ssp)
        return cls(helper, lib_ssp, dust_attenuation, dust_emission, selector_disk, selector_bulge)


    def forward(self, *args, return_ph=True):
        return self.generate(*self.adapter(*args), return_ph)


    def generate(self, gp, sfh_disk, sfh_bulge, return_ph=True):
        l_main = self.dust_attenuation(gp, sfh_disk, sfh_bulge)
        l_dust_slice, frac = self.dust_emission(gp, sfh_disk, sfh_bulge)
        l_dust = self.helper.set_item(torch.zeros_like(l_main), 'slice_lam_de', l_dust_slice)
        l_norm = self.helper.get_recover(gp, 'l_norm', torch)[:, None]
        l_tot = l_norm*(l_main + frac*l_dust)
        return torch.squeeze(self.detector(l_tot, return_ph))


    def configure_input_mode(self,
        sfr_bins=None, met_type='discrete', bounds_transform=True,
        simplex_transform=True, transform=None, fixed_params=None, device='cpu',
    ):
        self.adapter.configure(
            helper=self.helper,
            lib_ssp=self.lib_ssp,
            sfr_bins=sfr_bins,
            met_type=met_type,
            bounds_transform=bounds_transform,
            simplex_transform=simplex_transform,
            transform=transform,
            fixed_params=fixed_params,
            device=device,
        )


    def configure_output_mode(self, filters=None, z=0., distmod=0., ab_mag=False):
        self.detector.configure(filters=filters, z=z, distmod=distmod, ab_mag=ab_mag)


    @property
    def input_size(self):
        return self.adapter.input_size


    @property
    def bounds(self):
        return self.adapter.bounds


    @property
    def lam_pivot(self):
        """Pivot wavelength of the filters. [micrometer]"""
        return self.detector.lam_pivot


    @property
    def lam(self):
        """Wavelength of the output SED. [micrometer]"""
        return self.detector.lam


class Adapter(nn.Module):
    """Apply different parametrisation to input parameters"""
    def __init__(self, helper, lib_ssp):
        super().__init__()
        self.n_gp = len(helper.header)
        self._set_log_met(lib_ssp)
        self.configure(helper, lib_ssp)


    def forward(self, params):
        free_params = self.derive_free_params(params)
        model_params = self.derive_model_params(free_params)
        return model_params


    def configure(self,
        helper, lib_ssp, sfr_bins=None, met_type='discrete', bounds_transform=True,
        simplex_transform=True, transform=None, fixed_params=None, device='cpu'
    ):
        self.device = device
        self.bounds_transform = bounds_transform
        self.simplex_transform = simplex_transform
        self.transform = transform
        self._set_fixed_params(helper, fixed_params)
        self._set_free_shape(lib_ssp, sfr_bins, met_type)


    def derive_free_params(self, params, check_bounds=False):
        def simplex_transform(x):
            """Transform a hypercube into a simplex."""
            x = -torch.log(x)
            x = x/x.sum(dim=-1)[:, None]
            return x

        params = torch.as_tensor(params, dtype=torch.float32, device=self.device)
        if check_bounds:
            dim = params.dim() - 1
            is_out = torch.any(params <= self.lbounds, dim=dim) \
            | torch.any(params >= self.ubounds, dim=dim)

        if self.bounds_transform:
            eps = 1e-6
            params = F.hardtanh(params, eps, 1 - eps)
            params = (self._ub - self._lb)*params + self._lb

        free_params = self.unflatten(params)
        if self.simplex_transform:
            if self.n_free_sfh > 0:
                free_params[1] = simplex_transform(free_params[1])
            if self.n_free_sfh == 2:
                free_params[2] = simplex_transform(free_params[2])

        if self.transform is not None:
            free_params = self.transform(*free_params)
            assert(type(free_params) is tuple)

        if check_bounds:
            return free_params, is_out
        else:
            return free_params


    def derive_model_params(self, free_params):
        # Set fixed galaxy parameters
        gp_in = free_params[0]
        n_in = gp_in.size(0)
        gp = torch.zeros([n_in, self.n_gp], dtype=gp_in.dtype, device=gp_in.device)
        gp[:, self.fixed_inds] = self.fixed_gp
        gp[:, self.free_inds] = gp_in
        # Set fixed star formation histories
        if self.n_free_sfh == 0:
            sfh_disk = self.sfh_disk_fixed.tile((n_in, 1))
            sfh_bulge = self.sfh_bulge_fixed.tile((n_in, 1))
        elif self.n_free_sfh == 1:
            if 'sfh_disk' in self.fixed_sfh:
                sfh_disk = self.sfh_disk_fixed.tile((n_in, 1))
                sfh_bulge = self.derive_sfh(*free_params[1:])
            if 'sfh_bulge' in self.fixed_sfh:
                sfh_bulge = self.sfh_bulge_fixed.tile((n_in, 1))
                sfh_disk = self.derive_sfh(*free_params[1:])
        elif self.n_free_sfh == 2:
            sfh_disk = self.derive_sfh(*free_params[1::2])
            sfh_bulge = self.derive_sfh(*free_params[2::2])
        #
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
        return x_out

    
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
        sfr_out = torch.zeros([sfr.size(0), self.n_tau_ssp],
            dtype=sfr.dtype, layout=sfr.layout, device=sfr.device)
        for i_b, (idx_b, idx_e) in enumerate(self.sfr_bins):
            sfr_out[:, idx_b:idx_e] = sfr[:, i_b, None]/(idx_e - idx_b)
        return sfr_out


    def derive_idw_met(self, log_met):
        eps = 1e-6
        inv = 1./((log_met[:, None, :] - self.log_met)**2 + eps)
        weights = inv/inv.sum(dim=1)[:, None, :]
        return weights

    
    def _set_fixed_params(self, helper, fixed_params):
        if fixed_params is None:
            fixed_params = {}

        self.register_buffer('sfh_disk', torch.tensor(0., device=self.device))
        self.register_buffer('sfh_bulge', torch.tensor(0., device=self.device))

        n_free_sfh = 2
        fixed_inds = []
        fixed_gp = []
        fixed_sfh = []
        for key, val in fixed_params.items():
            if key in helper.header:
                fixed_inds.append(helper.lookup[key])
                fixed_gp.append(val)
            elif key == 'sfh_disk' or key == 'sfh_bulge':
                self.register_buffer(f"{key}_fixed", val)
                fixed_sfh.append(key)
                n_free_sfh -= 1
            else:
                KeyError(f"Unknown parameter: {key}.")
        self.n_free_sfh = n_free_sfh
        self.fixed_inds = tuple(fixed_inds)
        self.free_inds = tuple([i_p for i_p in range(self.n_gp) if i_p not in fixed_inds])
        self.fixed_sfh = fixed_sfh
        self.register_buffer("fixed_gp",
            torch.tensor(fixed_gp, dtype=torch.float32, device=self.device))


    def _set_free_shape(self, lib_ssp, sfr_bins, met_type):
        if sfr_bins is None:
            n_tau = lib_ssp.n_tau
        else:
            n_tau = len(sfr_bins)

        if met_type == 'discrete':
            n_met = lib_ssp.n_met
            dim_sfh = 1
        elif met_type == 'idw':
            n_met = lib_ssp.n_met
            dim_sfh = 2
        elif met_type == 'uni_idw':
            n_met = 1
            dim_sfh = 2
        else:
            raise ValueError(f"Unknown met_type: {met_type}.")

        if sfr_bins is not None and met_type == 'discrete':
            raise ValueError(f"SFR bins with discrete metallicity are not implemented.")
        
        if dim_sfh == 1:
            free_shape = tuple(
                [len(self.free_inds)] + [n_tau*n_met]*self.n_free_sfh
            )
            bounds = [(-1., 1.)]*len(self.free_inds) + [(0., 1.)]*n_tau*n_met*self.n_free_sfh
        else:
            free_shape = tuple(
                [len(self.free_inds)] + [n_tau]*self.n_free_sfh + [n_met]*self.n_free_sfh
            )
            bounds = [(-1., 1.)]*len(self.free_inds) + [(0., 1.)]*n_tau*self.n_free_sfh \
                + [(float(self.log_met[0]), float(self.log_met[-1]))]*n_met*self.n_free_sfh
        # Set attributes
        self.sfr_bins = sfr_bins
        self.met_type = met_type
        self.n_tau_ssp = lib_ssp.n_tau
        self.dim_sfh = dim_sfh
        self.free_shape = free_shape
        self.input_size = sum(free_shape)
        #
        if self.bounds_transform:
            lb, ub = torch.tensor(bounds, dtype=torch.float32, device=self.device).T
            self.register_buffer('_lb', lb)
            self.register_buffer('_ub', ub)
            self.bounds = [(-1, 1)]*self.input_size
        else:
            self.bounds = bounds
        lbounds, ubounds = torch.tensor(self.bounds, dtype=torch.float32, device=self.device).T
        self.register_buffer('lbounds', lbounds)
        self.register_buffer('ubounds', ubounds)


    def _set_log_met(self, lib_ssp):
        met_sol = 0.019
        self.register_buffer('log_met', torch.log10(lib_ssp.met/met_sol)[:, None])


class Detector(nn.Module):
    """Apply unit conversion and filters to the input fluxes.
    
    Parameters
    ----------
    filters : array
        An array of pyphot filter instances.
    lam : tensor [micrometer]
        Wavelength of the input fluxes.
    """
    def __init__(self, lam):
        super().__init__()
        self.register_buffer('lam', lam)
        self.lam_base = lam
        self.configure()
        self._set_unit()


    def forward(self, l_target, return_ph):
        """
        Parameters
        ----------
        l_target : tensor [?]
            Generalized flux density (nu*f_nu).
        """
        if return_ph:
            fluxes = torch.trapz(l_target[:, None, :]*self.trans_filter, self.lam)*self.dist_factor
            if self.ab_mag:
                return -2.5*torch.log10(fluxes) + 8.9
            else:
                return fluxes
        else:
            return l_target*self.lam*(self.unit_f_nu*self.dist_factor)


    def configure(self, filters=None, z=0., distmod=0., ab_mag=True):
        lam = self.lam_base*(1 + z)
        self.register_buffer('lam', lam)
        if filters is not None:
            trans_filter = np.zeros([len(filters), len(lam)])
            lam_pivot = np.zeros(len(filters))
            for i_f, ftr in enumerate(filters):
                trans_filter[i_f] = self._set_transmission(ftr, lam)
                lam_pivot[i_f] = self._set_lam_pivot(ftr)
            self.register_buffer('trans_filter', torch.tensor(trans_filter, dtype=torch.float32))
            self.register_buffer('lam_pivot', torch.tensor(lam_pivot, dtype=torch.float32))
        self.dist_factor = 10**(-.4*distmod)
        self.ab_mag = ab_mag


    def _set_transmission(self, ftr, lam):
        """The given flux should be in L_sol."""
        unit_lam = U.angstrom.to(U.micrometer)
        trans = np.interp(lam, ftr.wavelength*unit_lam, ftr.transmission, left=0., right=0.)
        trans = trans/np.trapz(trans/lam, lam)*self.unit_jansky
        return trans


    def _set_lam_pivot(self, ftr):
        return ftr.wave_pivot*U.angstrom.to(U.micrometer)


    def _set_unit(self):
        unit_f_nu = U.solLum/(4*np.pi*(10*U.parsec)**2)*U.micrometer/constants.c
        self.unit_f_nu = unit_f_nu.to(U.jansky).value
        self.unit_jansky = self.unit_f_nu


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

