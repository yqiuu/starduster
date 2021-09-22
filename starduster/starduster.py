from .utils import *
from .parameter import *
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


    def configure_input_mode(self, gp=None, sfh_disk=None, sfh_bulge=None, device='cpu'):
        self.adapter.configure(
            helper=self.helper, lib_ssp=self.lib_ssp,
            gp=gp, sfh_disk=sfh_disk, sfh_bulge=sfh_bulge,
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
        self.configure(helper, lib_ssp)


    def forward(self, params):
        params = torch.as_tensor(params, dtype=torch.float32, device=self.device)
        gp, sfh_disk, sfh_bulge, is_out = self.derive_free_params(params)
        return self.derive_model_params(gp, sfh_disk, sfh_bulge)


    def configure(self,
        helper, lib_ssp, gp=None, sfh_disk=None, sfh_bulge=None, device='cpu'
    ):
        if gp is None:
            gp = GalaxyParameter()
        if sfh_disk is None:
            sfh_disk = DiscreteSFH()
        if sfh_bulge is None:
            sfh_bulge = DiscreteSFH()
        self.pset_gp = gp.init(helper)
        self.pset_sfh_disk = sfh_disk.init(lib_ssp)
        self.pset_sfh_bulge = sfh_bulge.init(lib_ssp)
        self.device = device
        #
        pset_names = ['pset_gp', 'pset_sfh_disk', 'pset_sfh_bulge']
        free_shape = []
        bounds = []
        for name in pset_names:
            pset = getattr(self, name)
            free_shape.append(pset.input_size)
            bounds.append(pset.bounds)
        self.free_shape = free_shape
        self.input_size = sum(self.free_shape)
        #
        bounds = np.vstack(bounds)
        lbounds, ubounds = torch.tensor(bounds, dtype=torch.float32, device=self.device).T
        self.register_buffer('lbounds', lbounds)
        self.register_buffer('ubounds', ubounds)
        self.register_buffer('bound_radius', .5*(ubounds - lbounds))
        self.register_buffer('bound_centre', .5*(ubounds + lbounds))
        self.bounds = bounds
        self.bounds_transform = True


    def unflatten(self, params):
        params = torch.atleast_2d(params)
        params_out = [None]*len(self.free_shape)
        idx_b = 0
        for i_input, size in enumerate(self.free_shape):
            idx_e = idx_b + size
            params_out[i_input] = params[:, idx_b:idx_e]
            idx_b = idx_e
        return params_out


    def derive_free_params(self, params):
        dim = params.dim() - 1
        is_out = torch.any(params <= self.lbounds, dim=dim) \
            | torch.any(params >= self.ubounds, dim=dim)

        if self.bounds_transform:
            eps = 1e-6
            params = (params - self.bound_centre)/self.bound_radius
            params = F.hardtanh(params, -1 + eps, 1 - eps)
            params = self.bound_radius*params + self.bound_centre

        gp, sfh_disk, sfh_bulge = self.unflatten(params)
        return gp, sfh_disk, sfh_bulge, is_out


    def derive_model_params(self, gp, sfh_disk, sfh_bulge):
        return self.pset_gp(gp), self.pset_sfh_disk(sfh_disk), self.pset_sfh_bulge(sfh_bulge)


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
        self.n_ssp = self.n_met*self.n_tau 
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

