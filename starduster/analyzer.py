from .utils import units, accept_reject
from .starduster import MultiwavelengthSED
from .selector import sample_from_selector

import torch
import numpy as np


class Analyzer:
    def __init__(self, target):
        if isinstance(target, MultiwavelengthSED):
            self.sed_model = target
        else:
            self.posterior = target
            self.sed_model = target.sed_model
        self.helper = self.sed_model.helper
        self.lib_ssp = self.sed_model.adapter.lib_ssp


    def sample(self, n_samp=1, sampler=None, max_iter=10000):
        adapter = self.sed_model.adapter
        if not adapter.flat_input:
            raise ValueError("Set flat_input to be true.")

        if hasattr(self, "posterior"):
            n_col = self.posterior.input_size
            bounds = self.posterior.bounds
        else:
            n_col = self.sed_model.input_size
            bounds = self.sed_model.bounds
        lb, ub = torch.tensor(bounds, dtype=torch.float32).T
        if sampler is None:
            sampler = lambda n_samp: (ub - lb)*torch.rand([n_samp, n_col]) + lb

        def condition(params):
            cond = torch.all(params > lb, dim=-1) & torch.all(params < ub, dim=-1)
            gp = adapter(params)[0]
            cond &= adapter.selector_disk.select(self.helper.get_item(gp, 'curve_disk_inds')) \
                & adapter.selector_bulge.select(self.helper.get_item(gp, 'curve_bulge_inds'))
            return cond

        device = adapter.device
        try:
            adapter.cpu()
            samps = accept_reject(n_samp, n_col, sampler, condition, max_iter)
        finally:
            adapter.to(device)

        return samps


    def compute_parameter_summary(self, params, log_scale=False, print_summary=False):
        gp, sfh_disk, sfh_bulge = self.sed_model.adapter(params)
        gp_0 = self.helper.recover_all(gp, torch)

        theta = self.helper.get_item(gp_0, 'theta')
        r_disk = self.helper.get_item(gp_0, 'r_disk')
        r_bulge = self.helper.get_item(gp_0, 'r_bulge')
        r_dust = r_disk*self.helper.get_item(gp_0, 'r_dust_to_rd')
        l_norm = self.helper.get_item(gp_0, 'l_norm')
        b_to_t = self.helper.get_item(gp_0, 'b_to_t')
        m_dust = self.compute_m_dust(gp_0)
        m_disk, m_bulge = self.compute_m_star(gp_0, sfh_disk, sfh_bulge, separate=True)
        m_star = m_disk + m_bulge
        # SFR over 10 Myr
        sfr_10 = self.compute_sfr(gp_0, sfh_disk, sfh_bulge, time_scale=1e7, separate=False)
        # SFR over 100 Myr
        sfr_100 = self.compute_sfr(gp_0, sfh_disk, sfh_bulge, time_scale=1e8, separate=False)

        names = [
            'theta', 'r_disk', 'r_bulge', 'r_dust', 'l_norm', 'b_to_t',
            'm_dust', 'm_disk', 'm_bulge', 'm_star', 'sfr_10', 'sfr_100'
        ]
        variables = locals()
        summary = {}
        for key in names:
            if key == 'theta' or key == 'b_to_t':
                summary[key] = variables[key]
            else:
                if log_scale:
                    summary[key] = torch.log10(variables[key])
                else:
                    summary[key] = variables[key]

        if print_summary:
            for key, val in summary.items():
                msg = (key, val[0], getattr(units, key))
                if key == 'theta':
                    print("{}: {:.1f} {}".format(*msg))
                elif key == 'b_to_t':
                    print("{}: {:.2f} {}".format(*msg))
                else:
                    if log_scale:
                        print("{}: {:.2f} {}".format(*msg))
                    else:
                        print("{}: {:.2e} {}".format(*msg))

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


    def compute_metallicity(self, gp_0, sfh_disk, sfh_bulge, separate=False):
        def weighted_averge(arr, weights):
            return torch.sum(arr*weights, dim=-1)/torch.sum(weights, dim=-1)

        sfh_disk, sfh_bulge = self.convert_sfh(gp_0, sfh_disk, sfh_bulge)
        sfh_disk = self.lib_ssp.sum_over_age(sfh_disk)
        sfh_bulge = self.lib_ssp.sum_over_age(sfh_bulge)
        if separate:
            met_disk = weighted_averge(self.lib_ssp.met, sfh_disk)
            met_bulge = weighted_averge(self.lib_ssp.met, sfh_bulge)
            return met_disk, met_bulge
        else:
            return weighted_averge(self.lib_ssp.met, sfh_disk + sfh_bulge)


    def convert_sfh(self, gp_0, sfh_disk, sfh_bulge):
        convert = lambda sfh, l_norm: \
            self.lib_ssp.reshape_sfh(sfh)/self.lib_ssp.norm*l_norm[:, None, None]

        l_norm = self.helper.get_item(gp_0, 'l_norm')
        b_to_t = self.helper.get_item(gp_0, 'b_to_t')

        sfh_disk = convert(sfh_disk, l_norm*(1 - b_to_t))
        sfh_bulge = convert(sfh_bulge, l_norm*b_to_t)

        return sfh_disk, sfh_bulge

