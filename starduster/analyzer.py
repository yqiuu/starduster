from .utils import units, constants, accept_reject
from .sed_model import MultiwavelengthSED
from .selector import sample_from_selector

from functools import update_wrapper

import torch
import numpy as np


def register_calculator(name, input_type, is_separable):
    def wrapper(func):
        return update_wrapper(PropertyCalculator(func, name, input_type, is_separable), func)
    return wrapper


class PropertyCalculator:
    def __init__(self, func, name, input_type, is_separable):
        self.func = func
        self.name = name
        self.input_type = input_type
        self.is_separable = is_separable


    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class Analyzer:
    """Provide methods to calculate physical parameters.

    Parameters
    ----------
    target : MultiwavelengthSED or Posterior
        A MultiwavelengthSED or Posterior instance.
    """
    def __init__(self, sed_model):
        self.sed_model = sed_model
        self.helper = self.sed_model.helper
        self.lib_ssp = self.sed_model.adapter.lib_ssp
        self._collect_calculators()


    def list_available_properties(self):
        prop_names = list(self.helper.header)
        for name, calc in self._calculators.items():
            if calc.is_separable:
                prop_names.extend([name, name + '_disk', name + '_bulge'])
            else:
                prop_names.append(name)
        return prop_names


    def compute_property_summary(self, params, prop_names):
        gp_0, sfh_disk_0, sfh_bulge_0 = self.recover_params(params)
        output = {}
        for name in prop_names:
            if name in output:
                continue

            if name in self.helper.header:
                output[name] = self.helper.get_item(gp_0, name)
                continue

            if name.endswith('_disk'):
                name_disk = name
                name_bulge = name.replace('_disk', '_bulge')
                name = name.replace('_disk', '')
            elif name.endswith('_bulge'):
                name_disk = name.replace('_bulge', '_disk')
                name_bulge = name
                name = name.replace('_bulge', '')
            else:
                name_disk = None
                name_bulge = None

            calc = self._calculators[name]
            if calc.input_type == 'recovered_gp':
                args = gp_0,
            elif calc.input_type == 'recovered_params':
                args = gp_0, sfh_disk_0, sfh_bulge_0
            elif calc.input_type == 'scaled_params':
                args = params,
            else:
                raise ValueError

            if name_disk is None:
                output[name] = calc(self, *args)
            else:
                if calc.is_separable:
                    # Always store the properties of both the disk and bulge
                    # components. The unwanted property will be deleted later.
                    output[name_disk], output[name_bulge] = calc(self, *args, separate=True)
                else:
                    raise ValueError("{} is not separable.".format(name))

        for out_name in list(output.keys()):
            if out_name not in prop_names:
                del output[out_name]
        
        return output


    @register_calculator('frac_abs', 'scaled_params', is_separable=False)
    def compute_absorption_fraction(self, params):
        with torch.no_grad():
            frac = self.sed_model.predict_absorption_fraction(params)
        return frac


    @register_calculator('l_bol', 'scaled_params', is_separable=False)
    def compute_l_bol(self, params):
        """Compute bolometic luminosity.

        Parameters
        ----------
        params : tensor
            Scaled parameters.

        Returns
        -------
        l_bol : tensor [L_sol]
            Bolometric luminosity.
        """
        with torch.no_grad():
            lum = self.sed_model(params, return_ph=False, return_lum=True)
        return torch.trapz(lum, torch.log(self.sed_model.lam))


    @register_calculator('m_dust', 'recovered_gp', is_separable=False)
    def compute_m_dust(self, gp_0):
        """Compute dust mass.

        Parameters
        ----------
        gp_0 : tensor
            Recovered galaxy parameters.

        Returns
        ------
        m_dust : tensor [M_sol]
            Dust mass.
        """
        r_disk = self.helper.get_item(gp_0, 'r_disk')
        r_dust_to_rd = self.helper.get_item(gp_0, 'r_dust_to_rd')
        r_dust = r_disk*r_dust_to_rd
        den_dust = self.helper.get_item(gp_0, 'den_dust')
        return den_dust*(2*np.pi*r_dust*r_dust)


    @register_calculator('m_star', 'recovered_params', is_separable=True)
    def compute_m_star(self, gp_0, sfh_disk_0, sfh_bulge_0, separate=False):
        """Compute stellar mass.

        Parameters
        ----------
        gp_0 : tensor
            Recovered galaxy parameters.
        sfh_disk_0 : tensor [M_sol]
            Recovered SFH parameters of the disk component.
        sfh_bulge_0: tensor [M_sol]
            Recovered SFH parameters of the bulge component.
        separate : bool
            If True, return the properties of the disk and bulge components
            separately; otherwise return the total value.

        Returns
        -------
        tensor or (tensor, tensor) [M_sol]
            Stellar mass.
        """
        m_disk = sfh_disk_0.sum(dim=(1, 2))
        m_bulge = sfh_bulge_0.sum(dim=(1, 2))
        if separate:
            return m_disk, m_bulge
        else:
            return m_disk + m_bulge


    @register_calculator('sfr', 'recovered_params', is_separable=True)
    def compute_sfr(self, gp_0, sfh_disk_0, sfh_bulge_0, time_scale=1e8, separate=False):
        """Compute SFR.

        Parameters
        ----------
        gp_0 : tensor
            Recovered galaxy parameters.
        sfh_disk_0 : tensor [M_sol]
            Recovered SFH parameters of the disk component.
        sfh_bulge_0: tensor [M_sol]
            Recovered SFH parameters of the bulge component.
        time_scale : float [yr]
            Time scale of the SFR.
        separate : bool
            If True, return the properties of the disk and bulge components
            separately; otherwise return the total value.

        Returns
        -------
        tensor or (tensor, tensor) [M_sol]
            SFR.
        """
        sfh_disk_0 = self.lib_ssp.sum_over_met(sfh_disk_0)
        sfh_bulge_0 = self.lib_ssp.sum_over_met(sfh_bulge_0)

        d_tau = torch.cumsum(self.lib_ssp.d_tau, dim=0)
        idx = torch.argmin(torch.abs(d_tau - time_scale))
        sfr_disk = sfh_disk_0[:, :idx+1].sum(dim=1)/d_tau[idx]
        sfr_bulge = sfh_bulge_0[:, :idx+1].sum(dim=1)/d_tau[idx]

        if separate:
            return sfr_disk, sfr_bulge
        else:
            return sfr_disk + sfr_bulge


    @register_calculator('mwa', 'recovered_params', is_separable=True)
    def compute_mass_weighted_age(self, gp_0, sfh_disk_0, sfh_bulge_0, separate=False):
        """Compute mass weighted age.

        Parameters
        ----------
        gp_0 : tensor
            Recovered galaxy parameters.
        sfh_disk_0 : tensor [M_sol]
            Recovered SFH parameters of the disk component.
        sfh_bulge_0: tensor [M_sol]
            Recovered SFH parameters of the bulge component.
        separate : bool
            If True, return the properties of the disk and bulge components
            separately; otherwise return the total value.

        Returns
        -------
        tensor or (tensor, tensor) [M_sol]
            Mass weighted age.
        """
        def compute_average(tau, sfh):
            return torch.sum(tau*sfh, dim=-1)/torch.sum(sfh, dim=-1)
        
        tau = self.lib_ssp.tau
        sfh_disk_0 = self.lib_ssp.sum_over_met(sfh_disk_0)
        sfh_bulge_0 = self.lib_ssp.sum_over_met(sfh_bulge_0)
        if separate:
            tau_disk = compute_average(tau, sfh_disk_0)
            tau_bulge = compute_average(tau, sfh_bulge_0)
            return tau_disk, tau_bulge
        else:
            return compute_average(tau, sfh_disk_0 + sfh_bulge_0)


    @register_calculator('mwm', 'recovered_params', is_separable=True)
    def compute_mass_weighted_met(self, gp_0, sfh_disk_0, sfh_bulge_0, separate=False):
        """Compute mass weighted metallicity.

        Parameters
        ----------
        gp_0 : tensor
            Recovered galaxy parameters.
        sfh_disk_0 : tensor [M_sol]
            Recovered SFH parameters of the disk component.
        sfh_bulge_0: tensor [M_sol]
            Recovered SFH parameters of the bulge component.
        separate : bool
            If True, return the properties of the disk and bulge components
            separately; otherwise return the total value.

        Returns
        -------
        tensor or (tensor, tensor) [M_sol]
            Mass weighted metallicity.
        """
        def compute_average(met, sfh):
            met_mean = torch.sum(met*sfh, dim=-1)/torch.sum(sfh, dim=-1)
            return met_mean/constants.met_sol

        sfh_disk_0 = self.lib_ssp.sum_over_age(sfh_disk_0)
        sfh_bulge_0 = self.lib_ssp.sum_over_age(sfh_bulge_0)
        if separate:
            met_disk = compute_average(self.lib_ssp.met, sfh_disk_0)
            met_bulge = compute_average(self.lib_ssp.met, sfh_bulge_0)
            return met_disk, met_bulge
        else:
            return compute_average(self.lib_ssp.met, sfh_disk_0 + sfh_bulge_0)


    def recover_params(self, params, recover_sfh=True):
        """Transform the scaled parameters into physical parameters.

        Parameters
        ----------
        params : tensor
            Parameters that can be passed to a SED model.
        recover_sfh : bool
            If True, transform the SFH parameters into the mass in each SSP
            bins; otherwise remain the SFH parameters untransformed.

        Returns
        -------
        gp_0 : tensor
            Recovered galaxy parameters
        sfh_disk_0 : tensor [M_sol]
            (Recovered) SFH parameters of the disk component.
        sfh_bulge_0: tensor [M_sol]
            (Recovered) SFH parameters of the bulge component.
        """
        gp, sfh_disk, sfh_bulge = self.sed_model.adapter(params)
        gp_0 = self.helper.recover_all(gp, torch)
        if recover_sfh:
            sfh_disk_0, sfh_bulge_0 = self.recover_sfh(gp_0, sfh_disk, sfh_bulge)
            return gp_0, sfh_disk_0, sfh_bulge_0
        else:
            return gp_0, sfh_disk, sfh_bulge


    def recover_sfh(self, gp_0, sfh_disk, sfh_bulge):
        """Transform the SFH parameters into the mass in each SSP bins.

        Parameters
        ----------
        gp_0 : tensor
            Recovered galaxy parameters
        sfh_disk : tensor
            Scaled SFH parameters of the disk component.
        sfh_bulge : tensor
            Scaled SFH parameters of the bulge component.

        Returns
        -------
        sfh_disk_0 : tensor [M_sol]
            Recovered SFH parameters of the disk component.
        sfh_bulge_0: tensor [M_sol]
            Recovered SFH parameters of the bulge component.
        """
        def recover(sfh, l_norm):
            return self.lib_ssp.reshape_sfh(sfh)/self.lib_ssp.norm*l_norm[:, None, None]

        l_norm = self.helper.get_item(gp_0, 'l_norm')
        b_to_t = self.helper.get_item(gp_0, 'b_to_t')

        sfh_disk_0 = recover(sfh_disk, l_norm*(1 - b_to_t))
        sfh_bulge_0 = recover(sfh_bulge, l_norm*b_to_t)

        return sfh_disk_0, sfh_bulge_0


    def _collect_calculators(self):
        calculators = {}
        for name in dir(self):
            attr = getattr(self, name)
            if isinstance(attr, PropertyCalculator):
                calculators[attr.name] = attr
        self._calculators = calculators

