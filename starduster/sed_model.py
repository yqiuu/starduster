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

import torch
from torch import nn


class MultiwavelengthSED(nn.Module):
    """High-level API for computing multiwavelength SEDs.

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
    """
    def __init__(self,
        helper, lib_ssp, dust_attenuation, dust_emission, selector_disk=None, selector_bulge=None,
    ):
        super().__init__()
        self.helper = helper
        self.lib_ssp = lib_ssp
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
        dirname = path.join(path.dirname(path.abspath(__file__)), "data")
        fname_da_disk = path.join(dirname, "curve_disk.pt")
        fname_da_bulge = path.join(dirname, "curve_bulge.pt")
        fname_de = path.join(dirname, "emission.pt")
        fname_selector_disk = path.join(dirname, "selector_disk.pt")
        fname_selector_bulge = path.join(dirname, "selector_bulge.pt")
        return cls.from_checkpoint(
            lib_ssp, fname_da_disk, fname_da_bulge, fname_de,
            fname_selector_disk, fname_selector_bulge, map_location=torch.device('cpu')
        )


    def forward(self,
        *args, return_ph=False, return_lum=False, component='both', check_bounds=False,
    ):
        """Compute multi-wavelength SEDs.

        Parameters
        ----------
        gp : tensor
            Galaxy parameters.
        sfh_disk : tensor
            Star formation history of the disk component.
        sfh_bulge : tensor
            Star forward history of the bulge component.
        return_ph : bool
            If ``True``, apply filters to outputs.
        return_lum : bool
            If ``True``, return flux density in a unit of Jansky; otherwise
            return luminosity in a unit of L_sol.
        component : str {'both', 'star', 'dust'}
            'both' : Return SEDs including both stellar and dust emissions.
            'star' : Return stellar SEDs only.
            'dust' : Return dust SEDs only.
        check_bounds : bool
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
        if check_bounds:
            gp, sfh_disk, sfh_bulge, is_out = self.adapter(*args, check_bounds=check_bounds)
        else:
            gp, sfh_disk, sfh_bulge = self.adapter(*args, check_bounds=check_bounds)

        l_main = self.dust_attenuation(gp, sfh_disk, sfh_bulge)
        l_dust_slice, frac = self.dust_emission(gp, sfh_disk, sfh_bulge)
        l_dust = frac*self.helper.set_item(torch.zeros_like(l_main), 'slice_lam_de', l_dust_slice)
        l_norm = self.helper.get_recover(gp, 'l_norm', torch)[:, None]
        if component == 'both':
            l_ret = l_norm*(l_main + l_dust)
        elif component == 'star':
            l_ret = l_norm*l_main
        elif component == 'dust':
            l_ret = l_norm*l_dust
        else:
            raise ValueError(f"Unknow component: {component}.")
        l_ret = torch.squeeze(self.detector(l_ret, return_ph, return_lum))

        if check_bounds:
            return l_ret, is_out
        else:
            return l_ret

    
    def predict_absorption_fraction(self, *args):
        return torch.ravel(self.dust_emission(*self.adapter(*args))[-1])


    @wraps(Adapter.configure, assigned=('__doc__',))
    def configure_adapter(self, *args, **kwargs):
        self.adapter.configure(*args, **kwargs)

    
    @wraps(Detector.configure, assigned=('__doc__',))
    def configure_detector(self, *args, **kwargs):
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

