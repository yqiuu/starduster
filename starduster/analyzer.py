from .utils import units, constants, accept_reject
from .sed_model import MultiwavelengthSED
from .selector import sample_from_selector

import torch
import numpy as np


def register_calculator(name, input_type, is_separable):
    def add_attrs(func):
        func.name = name
        func.input_type = input_type
        func.is_separable = is_separable
        return func
    return add_attrs


class Analyzer:
    """Provide methods to calculate physical parameters.

    Parameters
    ----------
    sed_model : MultiwavelengthSED
        The SED model.
    """
    def __init__(self, sed_model):
        self.sed_model = sed_model
        self.helper = self.sed_model.helper
        self.lib_ssp = self.sed_model.adapter.lib_ssp
        self._collect_calculators()


    def list_available_properties(self):
        """List properties that can be computed by this class.

        Returns
        -------
        prop_names : list
            Property names.
        """
        prop_names = list(self.helper.header)
        for name, calc in self._calculators.items():
            if calc.is_separable:
                prop_names.extend([name, name + '_disk', name + '_bulge'])
            else:
                prop_names.append(name)
        return prop_names


    def compute_property_summary(self, params, prop_names, output_type='numpy'):
        """Compute properties given in the name list.

        Parameters
        ----------
        params : tensor
            Scaled parameters.
        prop_names : list
            Property names.

        Returns
        -------
        summary : dict
            Properties.
        """
        gp_0, sfh_disk_0, sfh_bulge_0 = self.recover_params(params)
        summary = {}
        for name in prop_names:
            if name in self.helper.header:
                summary[name] = self.helper.get_item(gp_0, name)
                continue

            if name in summary:
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

            try:
                calc = self._calculators[name]
            except KeyError:
                if name_disk is None:
                    raise ValueError(f"No calculator for '{name}'.")
                else:
                    raise ValueError(f"No calculator for '{name_disk}' and '{name_bulge}'.")

            if calc.input_type == 'recovered_gp':
                args = gp_0,
            elif calc.input_type == 'recovered_params':
                args = gp_0, sfh_disk_0, sfh_bulge_0
            elif calc.input_type == 'scaled_params':
                args = params,
            else:
                raise ValueError(f"Unknown input_type: '{calc.input_type}'.")

            with torch.no_grad():
                if name_disk is None:
                    summary[name] = calc(*args)
                else:
                    # Always store the properties of both the disk and bulge
                    # components. The unwanted property will be deleted later.
                    summary[name_disk], summary[name_bulge] = calc(*args, separate=True)

        for out_name in list(summary.keys()):
            if out_name not in prop_names:
                del summary[out_name]
            else:
                if output_type == 'numpy':
                    summary[out_name] = summary[out_name].cpu().numpy()
                elif output_type == 'torch':
                    pass
                else:
                    raise ValueError(f"Unknown output_type: '{output_type}'.")
        
        return summary


    @register_calculator('frac_abs', 'scaled_params', is_separable=False)
    def compute_absorption_fraction(self, params):
        """Compute dust absorption fraction.

        Parameters
        ----------
        params : tensor
            Scaled parameters.

        Returns
        -------
        frac : tensor
            Dust absorption fraction.
        """
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


    @register_calculator('r_dust', 'recovered_gp', is_separable=False)
    def compute_r_dust(self, gp_0):
        """Compute dust disk radius.

        Parameters
        ----------
        gp_0 : tensor
            Recovered galaxy parameters.

        Returns
        ------
        r_dust : tensor [kpc]
            Dust disk radius.
        """
        r_disk = self.helper.get_item(gp_0, 'r_disk')
        r_dust_to_rd = self.helper.get_item(gp_0, 'r_dust_to_rd')
        r_dust = r_disk*r_dust_to_rd
        return r_dust


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
        r_dust = self.compute_r_dust(gp_0)
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
        tau_edges = self.lib_ssp.tau_edges
        if time_scale < tau_edges[0] or time_scale > tau_edges[-1]:
            raise ValueError("Invalid time scale.")

        sfh_disk_0 = self.lib_ssp.sum_over_met(sfh_disk_0)
        sfh_bulge_0 = self.lib_ssp.sum_over_met(sfh_bulge_0)
        sfr_disk = torch.zeros(len(sfh_disk_0))
        sfr_bulge = torch.zeros(len(sfh_bulge_0))
        for i_bin in range(len(tau_edges)):
            if time_scale >= tau_edges[i_bin + 1]:
                sfr_disk += sfh_disk_0[:, i_bin]
                sfr_bulge += sfh_bulge_0[:, i_bin]
            else:
                frac = (time_scale - tau_edges[i_bin])/(tau_edges[i_bin+1] - tau_edges[i_bin])
                sfr_disk += sfh_disk_0[:, i_bin]*frac
                break
        sfr_disk /= time_scale
        sfr_bulge /= time_scale

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
        tensor or (tensor, tensor) [Z_sol]
            Mass weighted metallicity.
        """
        def compute_average(met, sfh):
            met_mean = torch.sum(met*sfh, dim=-1)/torch.sum(sfh, dim=-1)
            return met_mean/constants.met_sol

        sfh_disk_0 = self.lib_ssp.sum_over_age(sfh_disk_0)
        sfh_bulge_0 = self.lib_ssp.sum_over_age(sfh_bulge_0)
        met = self.lib_ssp.met.clone().to(gp_0.device)
        if separate:
            met_disk = compute_average(met, sfh_disk_0)
            met_bulge = compute_average(met, sfh_bulge_0)
            return met_disk, met_bulge
        else:
            return compute_average(met, sfh_disk_0 + sfh_bulge_0)


    @register_calculator('sfh', 'recovered_params', is_separable=True)
    def compute_sfh(self, gp_0, sfh_disk_0, sfh_bulge_0, separate=False):
        """Compute star formation history.

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
            Stellar mass in each time bin.
        """
        sfh_disk_0 = self.lib_ssp.sum_over_met(sfh_disk_0)
        sfh_bulge_0 = self.lib_ssp.sum_over_met(sfh_bulge_0)
        if separate:
            return sfh_disk_0, sfh_bulge_0
        else:
            return sfh_disk_0 + sfh_bulge_0


    @register_calculator('mh', 'recovered_params', is_separable=True)
    def compute_mh(self, gp_0, sfh_disk_0, sfh_bulge_0, separate=False):
        """Compute mass weighted metallicity history.

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
        tensor or (tensor, tensor) [Z_sol]
            Mass weighted metallicity history.
        """
        def compute_average(met, sfh):
            met_mean = torch.sum(met*sfh, dim=dim_met)/torch.sum(sfh, dim=dim_met)
            return met_mean/constants.met_sol

        dim_met = self.lib_ssp.dim_met
        sizes = [1, 1, 1]
        sizes[dim_met] = -1
        met = self.lib_ssp.met.unflatten(0, sizes).clone().to(gp_0.device)
        if separate:
            met_disk = compute_average(met, sfh_disk_0)
            met_bulge = compute_average(met, sfh_bulge_0)
            return met_disk, met_bulge
        else:
            return compute_average(met, sfh_disk_0 + sfh_bulge_0)


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
        with torch.no_grad():
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
            return self.lib_ssp.reshape_sfh(sfh)/l_ssp_norm*l_norm[:, None, None]

        l_ssp_norm = self.lib_ssp.norm.clone().to(gp_0.device)
        l_norm = self.helper.get_item(gp_0, 'l_norm')
        b_to_t = self.helper.get_item(gp_0, 'b_to_t')

        sfh_disk_0 = recover(sfh_disk, l_norm*(1 - b_to_t))
        sfh_bulge_0 = recover(sfh_bulge, l_norm*b_to_t)

        return sfh_disk_0, sfh_bulge_0


    def _collect_calculators(self):
        calculators = {}
        for name in dir(self):
            attr = getattr(self, name)
            if hasattr(attr, 'name'):
                calculators[attr.name] = attr
        self._calculators = calculators

