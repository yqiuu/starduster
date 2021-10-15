from .utils import *
from .parameter import *
from .lib_ssp import SSPLibrary
from .selector import Selector
from .dust_attenuation import AttenuationCurve, DustAttenuation
from .dust_emission import DustEmission

import pickle
from os import path
from functools import wraps

import numpy as np
from astropy import units as U
from astropy import constants
import torch
from torch import nn
from torch.nn import functional as F


class Adapter(nn.Module):
    """Apply different parametrisations to input parameters.

    Parameters
    ----------
    helper : Helper
        Helper of input parameters.
    lib_ssp : SSPLibrary
        A simple stellar population library.
    selector_disk : Selector
        Selector of the disk component.
    selector_bulge : Selector
        Selector of the bulge component.
    """
    def __init__(self, helper, lib_ssp, selector_disk=None, selector_bulge=None):
        super().__init__()
        self.helper = helper
        self.lib_ssp = lib_ssp
        if selector_disk is not None:
            self.selector_disk = selector_disk
        if selector_bulge is not None:
            self.selector_bulge = selector_bulge
        self.register_buffer("_device", torch.tensor(0.), persistent=False)
        self.configure()


    def forward(self, *args, check_bounds=False):
        if self.flat_input:
            params = torch.as_tensor(args[0], dtype=torch.float32, device=self.device)
            gp, sfh_disk, sfh_bulge = self.unflatten(params)
        else:
            gp, sfh_disk, sfh_bulge = args

        if check_bounds:
            is_out = torch.full((gp.size(0),), False, device=self.device)
            for val, pset in zip(
                [gp, sfh_disk, sfh_bulge],
                [self.pset_gp, self.pset_sfh_disk, self.pset_sfh_bulge]
            ):
                is_out |= pset.check_bounds(val)

            gp = self.pset_gp(gp)
            sfh_disk = self.pset_sfh_disk(sfh_disk)
            sfh_bulge = self.pset_sfh_bulge(sfh_bulge)

            ## Assume helper of selector_disk and selector_bulge are the same
            helper = self.selector_disk.helper
            is_out |= ~self.selector_disk.select(helper.get_item(gp, 'curve_disk_inds'))
            is_out |= ~self.selector_bulge.select(helper.get_item(gp, 'curve_bulge_inds'))

            return gp, sfh_disk, sfh_bulge, is_out
        else:
            return self.pset_gp(gp), self.pset_sfh_disk(sfh_disk), self.pset_sfh_bulge(sfh_bulge)


    def configure(self, gp=None, sfh_disk=None, sfh_bulge=None, flat_input=False):
        """Configure the input mode.

        Parameters
        ----------
        gp : ParameterSet
            Parametrisation of the galaxy parameters.
        sfh_disk : ParameterSet
            Parametrisation of the disk star formation history.
        sfh_bulge : ParameterSet
            Parametrisation of the bulge star formation history.
        flat_input : bool
            If True, assume the input array is flat.
        """
        if gp is None:
            gp = GalaxyParameter()
        if sfh_disk is None:
            sfh_disk = DiscreteSFH()
        if sfh_bulge is None:
            sfh_bulge = DiscreteSFH()
        self.pset_gp = gp.init(self.helper)
        self.pset_sfh_disk = sfh_disk.init(self.lib_ssp)
        self.pset_sfh_bulge = sfh_bulge.init(self.lib_ssp)
        self.flat_input = flat_input
        #
        pset_names = ['pset_gp', 'pset_sfh_disk', 'pset_sfh_bulge']
        free_shape = []
        param_names = []
        bounds = []
        for name in pset_names:
            pset = getattr(self, name)
            free_shape.append(pset.input_size)
            param_names.extend(pset.param_names)
            bounds.append(pset.bounds)
        self.free_shape = free_shape
        self.input_size = sum(self.free_shape)
        self.param_names = param_names
        self.bounds = np.vstack(bounds)


    def unflatten(self, params):
        params = torch.atleast_2d(params)
        params_out = [None]*len(self.free_shape)
        idx_b = 0
        for i_input, size in enumerate(self.free_shape):
            idx_e = idx_b + size
            params_out[i_input] = params[:, idx_b:idx_e]
            idx_b = idx_e
        return params_out


    @property
    def device(self):
        return self._device.device


class Detector(nn.Module):
    """Apply unit conversion and filters to the input fluxes.

    Parameters
    ----------
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
        ## The given flux should be in L_sol.
        if return_ph:
            return self.apply_filters(l_target)
        else:
            if self.jansky:
                return l_target*self.lam*(self.unit_f_nu*self.dist_factor)
            else:
                return l_target


    def apply_filters(self, l_target):
        ## The given flux should be in L_sol.
        fluxes = torch.trapz(l_target[:, None, :]*self.trans_filter, self.lam)*self.dist_factor
        if self.ab_mag:
            return -2.5*torch.log10(fluxes) + 8.9
        else:
            return fluxes


    def configure(self, filters=None, z=0., distmod=0., ab_mag=True, jansky=True):
        """Configure the output mode.

        Parameters
        ----------
        filters : sedpy.observate.Filter
            Output filters.
        z : float
            Redshift.
        distmod : float
            Distance modulus.
        ab_mag : bool
            If True, return AB magnitudes; otherwise return flux densities.
        jansky : bool
            If True, the unit of the output full spectra will be Jansky;
            otherwise the unit will be L_sol.
        """
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
        self.jansky = jansky


    def _set_transmission(self, ftr, lam):
        ## The given flux and wavelength should be in L_sol and micrometer
        ## respectively.
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


class MultiwavelengthSED(nn.Module):
    """Primary module to compute multiwavelength SEDs.

    Parameters
    ----------
    helper : Helper
        Helper of input parameters.
    dust_attenuation : module
        Dust attenuation module.
    dust_emission : module
        Dust emission module.
    selector_disk : Selector
        Selector of the disk component.
    selector_bulge : Selector
        Selector of the bulge component.

    Inputs
    ------
    gp : tensor
        Galaxy parameters.
    sfh_disk : tensor
        Star formation history of the disk component.
    sfh_bulge : tensor
        Star forward history of the bulge component.
    return_ph : bool
        If True, apply filters to outputs.

    Outputs
    -------
        If return_ph is true, return filter fluxes; otherwise return full
        spectra.
    """
    def __init__(self,
        helper, lib_ssp, dust_attenuation, dust_emission, selector_disk=None, selector_bulge=None,
    ):
        super().__init__()
        self.helper = helper
        self.dust_attenuation = dust_attenuation
        self.dust_emission = dust_emission
        self.adapter = Adapter(helper, lib_ssp, selector_disk, selector_bulge)
        self.detector = Detector(lib_ssp.lam)


    @classmethod
    def from_checkpoint(
        cls, lib_ssp, fname_da_disk, fname_da_bulge, fname_de,
        fname_selector_disk, fname_selector_bulge, map_location=None
    ):
        """Initialise an instance from a checkpoint.

        Parameters
        ----------
        lib_ssp : SSPLibrary
            A simple stellar population library.
        fname_da_disk : str
            File name of the disk attenuation curve module.
        fname_da_bulge : str
            File name of the bulge attenuation curve module.
        fname_de : str
            File name of the dust emission module.
        fname_selector_disk : str
            File name of the selector corresponding to the disk component.
        fname_selector_bulge : str
            File name of the selector corresponding to the bugle component.
        map_location
            A variable that is passed to torch.load.
        """
        curve_disk, _ = load_model(fname_da_disk, AttenuationCurve, map_location=map_location)
        curve_bulge, _ = load_model(fname_da_bulge, AttenuationCurve, map_location=map_location)
        selector_disk, _ = load_model(fname_selector_disk, Selector, map_location=map_location)
        selector_bulge, _ = load_model(fname_selector_bulge, Selector, map_location=map_location)
        dust_emission = \
            DustEmission.from_checkpoint(fname_de, lib_ssp.L_ssp, map_location=map_location)
        helper = dust_emission.helper
        dust_attenuation = DustAttenuation(helper, curve_disk, curve_bulge, lib_ssp.l_ssp)
        return cls(helper, lib_ssp, dust_attenuation, dust_emission, selector_disk, selector_bulge)


    @classmethod
    def from_builtin(cls):
        """Initialise the built-in SED model."""
        lib_ssp = SSPLibrary.from_builtin()
        dirname = path.join(path.dirname(path.abspath(__file__)), "models")
        fname_da_disk = path.join(dirname, "curve_disk.pt")
        fname_da_bulge = path.join(dirname, "curve_bulge.pt")
        fname_de = path.join(dirname, "emission.pt")
        fname_selector_disk = path.join(dirname, "selector_disk.pt")
        fname_selector_bulge = path.join(dirname, "selector_bulge.pt")
        return cls.from_checkpoint(
            lib_ssp, fname_da_disk, fname_da_bulge, fname_de,
            fname_selector_disk, fname_selector_bulge, map_location=torch.device('cpu')
        )


    def forward(self, *args, return_ph=False, check_bounds=False):
        if check_bounds:
            gp, sfh_disk, sfh_bulge, is_out = self.adapter(*args, check_bounds=check_bounds)
        else:
            gp, sfh_disk, sfh_bulge = self.adapter(*args, check_bounds=check_bounds)

        l_main = self.dust_attenuation(gp, sfh_disk, sfh_bulge)
        l_dust_slice, frac = self.dust_emission(gp, sfh_disk, sfh_bulge)
        l_dust = self.helper.set_item(torch.zeros_like(l_main), 'slice_lam_de', l_dust_slice)
        l_norm = self.helper.get_recover(gp, 'l_norm', torch)[:, None]
        l_tot = l_norm*(l_main + frac*l_dust)
        l_ret = torch.squeeze(self.detector(l_tot, return_ph))

        if check_bounds:
            return l_ret, is_out
        else:
            return l_ret


    @wraps(Adapter.configure, assigned=('__doc__',))
    def configure_input_mode(self, *args, **kwargs):
        self.adapter.configure(*args, **kwargs)

    
    @wraps(Detector.configure, assigned=('__doc__',))
    def configure_output_mode(self, *args, **kwargs):
        self.detector.configure(*args, **kwargs)


    @property
    def input_size(self):
        """Number of input parameters."""
        return self.adapter.input_size


    @property
    def param_names(self):
        """Parameter names"""
        return self.adapter.param_names


    @property
    def bounds(self):
        """Bounds of input parameters."""
        return self.adapter.bounds


    @property
    def lam_pivot(self):
        """Pivot wavelength of the filters. [micrometer]"""
        return self.detector.lam_pivot


    @property
    def lam(self):
        """Wavelength of the SED. [micrometer]"""
        return self.detector.lam

