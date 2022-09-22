from .utils import *
from .adapter import Adapter
from .detector import Detector
from .lib_ssp import SSPLibrary
from .selector import Selector
from .dust_attenuation import AttenuationCurve, DustAttenuation
from .dust_emission import DustEmission

import pickle
from os import path
from functools import wraps

import numpy as np
import torch
from torch import nn


class MultiwavelengthSED(nn.Module):
    """High-level API for computing multiwavelength SEDs.

    Parameters
    ----------
    helper : Helper
        Helper of input parameters.
    curve_disk : module
        Attenuation curve module for the stellar disk.
    curve_bulge : module
        Attenuation curve module for the stellar bulge.
    dust_emission : module
        Dust emission module.
    selector_disk : Selector
        Selector of the disk component.
    selector_bulge : Selector
        Selector of the bulge component.
    """
    def __init__(self,
        helper, lib_ssp, curve_disk, curve_bulge, dust_emission,
        selector_disk=None, selector_bulge=None,
    ):
        super().__init__()
        self.helper = helper
        self.lib_ssp = lib_ssp
        interp_da, interp_de = self._prepare_regrid_modules(helper, lib_ssp)
        self.dust_attenuation = \
            DustAttenuation(helper, curve_disk, curve_bulge, lib_ssp.l_ssp, interp_da)
        self.dust_emission = dust_emission
        self.interp_de = interp_de
        self.adapter = Adapter(helper, lib_ssp, selector_disk, selector_bulge)
        self.detector = Detector(lib_ssp.lam_eval)


    def _prepare_regrid_modules(self, helper, lib_ssp):
        """Prepare regrid modules, which change the wavelength grid of the dust
        attenuation and emission modules.
        """
        if lib_ssp.regrid == 'base':
            interp_da = None
            interp_de = None
        else:
            lam_da = lib_ssp.lam_base[helper.lookup[f'slice_lam_da']]
            interp_da = Regrid(lib_ssp.lam_eval, lam_da, 1.)
            lam_de = lib_ssp.lam_base[helper.lookup[f'slice_lam_de']]
            interp_de = Regrid(lib_ssp.lam_eval, lam_de, 0.)
        return interp_da, interp_de


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
        return cls(
            helper, lib_ssp, curve_disk, curve_bulge, dust_emission, selector_disk, selector_bulge
        )


    @classmethod
    def from_builtin(cls, regrid='auto'):
        """Initialise the built-in SED model.
        
        Parameters
        ----------
        regrid : str {'base', 'auto', 'full'} or array
            Specify the wavelength grid for the SED model.

            'base': the grid of training data.
            'auto': a grid which includes some line emission features.
            'full': the original grid of the SSP library.

            The wavelength grid can also be given by an array.
        """
        map_location = torch.device('cpu')
        dirname = path.join(path.dirname(path.abspath(__file__)), "data")
        fname_da_disk = path.join(dirname, "curve_disk.pt")
        fname_da_bulge = path.join(dirname, "curve_bulge.pt")
        fname_de = path.join(dirname, "dust_emission_v1.pt")
        fname_selector_disk = path.join(dirname, "selector_disk.pt")
        fname_selector_bulge = path.join(dirname, "selector_bulge.pt")
        eps_reduce = torch.load(fname_de, map_location=map_location)['eps_reduce']
        lib_ssp = SSPLibrary.from_builtin(regrid, eps_reduce)
        return cls.from_checkpoint(
            lib_ssp, fname_da_disk, fname_da_bulge, fname_de,
            fname_selector_disk, fname_selector_bulge, map_location=map_location
        )


    def forward(self,
        *args, return_ph=False, return_lum=False, component='combine', check_selector=False,
    ):
        """Compute multi-wavelength SEDs.

        Parameters
        ----------
        gp : tensor
            Galaxy parameters.
        sfh_disk : tensor
            Star formation history of the disk component.
        sfh_bulge : tensor
            Star formation history of the bulge component.
        return_ph : bool
            If ``True``, apply filters to outputs.
        return_lum : bool
            If ``True``, return luminosity in a unit of L_sol; otherwise return
            flux density in a unit of Jansky.
        component : str 
            'combine' : Return SEDs including both stellar and dust emissions.
            'dust_free': Return dust free SEDs.
            'dust_attenuation' : Return dust attenuated stellar SEDs only.
            'dust_emission' : Return dust SEDs only.
        check_selector : bool
            If ``True``, return an additional tensor indicating whether the
            input parameters are beyond the effective region.

        Returns
        -------
        l_ret : tensor
            Multi-wavelength SEDs. The actual values depends on ``return_ph``,
            ``return_lum`` and ``component``.
        is_out : tensor
            A boolean array of where the input parameters are beyond the
            effective region.
        """
        if check_selector:
            gp, sfh_disk, sfh_bulge, log_p_in = self.adapter(*args, check_selector=check_selector)
        else:
            gp, sfh_disk, sfh_bulge = self.adapter(*args, check_selector=check_selector)

        if component == 'dust_free':
            apply_dust = False
        else:
            apply_dust = True
        l_main = self.dust_attenuation(gp, sfh_disk, sfh_bulge, apply_dust)

        l_dust_slice, frac = self.dust_emission(gp, sfh_disk, sfh_bulge)
        if self.interp_de is None:
            l_dust = self.helper.set_item(torch.zeros_like(l_main), 'slice_lam_de', l_dust_slice)
        else:
            l_dust = self.interp_de(l_dust_slice)
        l_dust = frac*l_dust

        l_norm = self.helper.get_recover(gp, 'l_norm', torch)[:, None]
        if component == 'combine':
            l_ret = l_norm*(l_main + l_dust)
        elif component == 'dust_free' or component == 'dust_attenuation':
            l_ret = l_norm*l_main
        elif component == 'dust_emission':
            l_ret = l_norm*l_dust
        else:
            raise ValueError(f"Unknow component: {component}.")
        l_ret = torch.squeeze(self.detector(l_ret, return_ph, return_lum))

        if check_selector:
            return l_ret, log_p_in
        else:
            return l_ret

    
    def predict_absorption_fraction(self, *args, return_lum=False):
        gp, sfh_disk, sfh_bulge = self.adapter(*args)
        frac = torch.ravel(self.dust_emission(gp, sfh_disk, sfh_bulge)[-1])
        if return_lum:
            return frac*self.helper.get_recover(gp, 'l_norm')
        else:
            return frac


    def predict_attenuation(self, *args, windows):
        filters_0 = self.detector.filters
        filters = [[np.array([lam_l, lam_u]), np.ones(2)] for lam_l, lam_u in windows]
        try:
            self.configure(filters=filters, ab_mag=True)
            mags_no_dust = self(*args, return_ph=True, component='dust_free')
            mags_dust = self(*args, return_ph=True, component='combine')
        finally:
            self.configure(filters=filters_0)
        return mags_dust - mags_no_dust


    def configure(self, **kwargs):
        """Configure the input and output settings.

        Parameters
        ----------
        pn_gp : ParameterSet
            Parametrisation of the galaxy parameters.
        pn_sfh_disk : ParameterSet
            Parametrisation of the disk star formation history.
        pn_sfh_bulge : ParameterSet
            Parametrisation of the bulge star formation history.
        flat_input : bool
            If ``True``, assume the input array is flat.
        check_sfh_norm : bool
            If ``True``, raise an error when star formation history is not
            normalised to one.
        filters : sedpy.observate.Filter
            Output filters.
        redshift : float
            Redshift.
        distmod : float
            Distance modulus.
        ab_mag : bool
            If ``True``, return AB magnitudes; otherwise return flux densities.
        """
        config_adapter = {}
        names_adapter = list(self.adapter.get_config())
        config_detector = {}
        names_detector = list(self.detector.get_config())
        for key, val in kwargs.items():
            if key in names_adapter:
                config_adapter[key] = val
            elif key in names_detector:
                config_detector[key] = val
            else:
                raise ValueError(f"Unknow config: {key}.")
        self.adapter.configure(**config_adapter)
        self.detector.configure(**config_detector)
        # The configure methods may register new buffers, the following line
        # makes their device consistent.
        self.to(self.adapter.device)


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

