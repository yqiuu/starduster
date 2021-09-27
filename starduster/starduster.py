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
        If return_ph is true, return filter fluxes. If return_ph is false,
        return full spectra.
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
        """Configure the input mode.

        Parameters
        ----------
        gp : ParameterSet
            Parametrisation of the galaxy parameters.
        sfh_disk : ParameterSet
            Parametrisation of the disk component star formation history.
        sfh_bulge : ParameterSet
            Parametrisation of the bulge component star formation history.
        device : torch.device
            Device of the module.
        """
        self.adapter.configure(
            helper=self.helper, lib_ssp=self.lib_ssp,
            gp=gp, sfh_disk=sfh_disk, sfh_bulge=sfh_bulge,
            device=device,
        )


    def configure_output_mode(self, filters=None, z=0., distmod=0., ab_mag=False):
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
            If True, return AB magnitudes. If False, return flux densities.
        """
        self.detector.configure(filters=filters, z=z, distmod=distmod, ab_mag=ab_mag)


    def compute_parameter_summary(self, params, print_summary=False):
        gp, sfh_disk, sfh_bulge = self.adapter(params)
        gp_0 = self.helper.recover_all(gp, torch)

        theta = self.helper.get_item(gp_0, 'theta')
        r_disk = self.helper.get_item(gp_0, 'r_disk')
        r_bulge = self.helper.get_item(gp_0, 'r_bulge')
        r_dust = r_disk*self.helper.get_item(gp_0, 'r_dust_to_rd')
        m_dust = self.compute_m_dust(gp_0)
        m_disk, m_bulge = self.compute_m_star(gp_0, sfh_disk, sfh_bulge, separate=True)
        m_star = m_disk + m_bulge
        # SFR over 10 Myr
        sfr_10 = self.compute_sfr(gp_0, sfh_disk, sfh_bulge, time_scale=1e7, separate=False)
        # SFR over 100 Myr
        sfr_100 = self.compute_sfr(gp_0, sfh_disk, sfh_bulge, time_scale=1e8, separate=False)

        names = [
            'theta', 'r_disk', 'r_bulge', 'r_dust', 'm_dust',
            'm_disk', 'm_bulge', 'm_star', 'sfr_10', 'sfr_100'
        ]
        # TODO: put units in somewhere more formal
        units = {
            'theta': '',
            'r_disk': 'kpc',
            'r_bulge': 'kpc',
            'r_dust': 'kpc',
            'm_dust': 'M_sol',
            'm_disk': 'M_sol',
            'm_bulge': 'M_sol',
            'm_star': 'M_sol',
            'sfr_10': 'M_sol/yr',
            'sfr_100': 'M_sol/yr'
        }
        variables = locals()
        summary = {key: variables[key] for key in names}

        if print_summary:
            for key, val in summary.items():
                if key == 'theta':
                    print("{}: {:.1f} {}".format(key, val[0], units[key]))
                else:
                    print("{}: {:.3e} {}".format(key, val[0], units[key]))

        return summary


    def compute_m_dust(self, gp_0):
        r_disk = self.helper.get_item(gp_0, 'r_disk')
        r_dust_to_rd = self.helper.get_item(gp_0, 'r_dust_to_rd')
        r_dust = r_disk*r_dust_to_rd
        den_dust = self.helper.get_item(gp_0, 'den_dust')
        return den_dust*(2*np.pi*r_dust*r_dust)


    def compute_m_star(self, gp_0, sfh_disk, sfh_bulge, separate=False):
        sfh_disk, sfh_bulge = self.convert_sfh(gp_0, sfh_disk, sfh_bulge)
        m_disk = sfh_disk.sum(dim=(1, 2))
        m_bulge = sfh_bulge.sum(dim=(1, 2))
        if separate:
            return m_disk, m_bulge
        else:
            return m_disk + m_bulge


    def compute_sfr(self, gp_0, sfh_disk, sfh_bulge, time_scale=1e8, separate=False):
        sfh_disk, sfh_bulge = self.convert_sfh(gp_0, sfh_disk, sfh_bulge)
        sfh_disk = self.lib_ssp.sum_over_met(sfh_disk)
        sfh_bulge = self.lib_ssp.sum_over_met(sfh_bulge)

        d_tau = torch.cumsum(self.lib_ssp.d_tau, dim=0)
        idx = torch.argmin(torch.abs(d_tau - time_scale))
        sfr_disk = sfh_disk[:, :idx+1].sum(dim=1)/d_tau[idx]
        sfr_bulge = sfh_bulge[:, :idx+1].sum(dim=1)/d_tau[idx]

        if separate:
            return sfr_disk, sfr_bulge
        else:
            return sfr_disk + sfr_bulge


    def convert_sfh(self, gp_0, sfh_disk, sfh_bulge):
        convert = lambda sfh, l_norm: self.lib_ssp.reshape_sfh(sfh)/self.lib_ssp.norm*l_norm

        l_norm = self.helper.get_item(gp_0, 'l_norm')
        b_to_t = self.helper.get_item(gp_0, 'b_to_t')

        sfh_disk = convert(sfh_disk, l_norm*(1 - b_to_t))
        sfh_bulge = convert(sfh_bulge, l_norm*b_to_t)

        return sfh_disk, sfh_bulge


    @property
    def input_size(self):
        """Number of input parameters."""
        return self.adapter.input_size


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

