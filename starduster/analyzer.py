from .utils import units, constants, accept_reject
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


    def recover(self, params):
        gp, sfh_disk, sfh_bulge = self.sed_model.adapter(params)
        gp_0 = self.helper.recover_all(gp, torch)
        sfh_disk_0, sfh_bulge_0 = self.recover_sfh(gp_0, sfh_disk, sfh_bulge)
        return gp_0, sfh_disk_0, sfh_bulge_0


    def compute_parameter_summary(self, params, log_scale=False, print_summary=False):
        gp_0, sfh_disk_0, sfh_bulge_0 = self.recover(params)

        theta = self.helper.get_item(gp_0, 'theta')
        r_disk = self.helper.get_item(gp_0, 'r_disk')
        r_bulge = self.helper.get_item(gp_0, 'r_bulge')
        r_dust = r_disk*self.helper.get_item(gp_0, 'r_dust_to_rd')
        l_norm = self.helper.get_item(gp_0, 'l_norm')
        b_to_t = self.helper.get_item(gp_0, 'b_to_t')
        m_dust = self.compute_m_dust(gp_0)
        m_disk, m_bulge = self.compute_m_star(gp_0, sfh_disk_0, sfh_bulge_0, separate=True)
        m_star = m_disk + m_bulge
        # SFR over 100 Myr
        sfr_disk, sfr_bulge = \
            self.compute_sfr(gp_0, sfh_disk_0, sfh_bulge_0, time_scale=1e8, separate=True)
        #
        met_disk, met_bulge = \
            self.compute_met(gp_0, sfh_disk_0, sfh_bulge_0, separate=True)

        names = [
            'theta', 'r_disk', 'r_bulge', 'r_dust', 'l_norm', 'b_to_t', 'm_dust', 'm_disk',
            'm_bulge', 'm_star', 'sfr_disk', 'sfr_bulge', 'met_disk', 'met_bulge'
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
                if 'sfr' in key:
                    unit_name = 'sfr'
                elif 'met' in key:
                    unit_name = 'met'
                else:
                    unit_name = key
                msg = (key, val[0], getattr(units, unit_name))
                if key == 'theta':
                    print("{}: {:.1f} {}".format(*msg))
                elif key == 'b_to_t':
                    print("{}: {:.2f} {}".format(*msg))
                elif 'met' in key:
                    print("{}: {:.3f} {}".format(*msg))
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


    def compute_m_star(self, gp_0, sfh_disk_0, sfh_bulge_0, separate=False):
        m_disk = sfh_disk_0.sum(dim=(1, 2))
        m_bulge = sfh_bulge_0.sum(dim=(1, 2))
        if separate:
            return m_disk, m_bulge
        else:
            return m_disk + m_bulge


    def compute_sfr(self, gp_0, sfh_disk_0, sfh_bulge_0, time_scale=1e8, separate=False):
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


    def compute_met(self, gp_0, sfh_disk_0, sfh_bulge_0, separate=False):
        def mass_weighted_metallicity(met, sfh):
            met_mean = torch.sum(met*sfh, dim=-1)/torch.sum(sfh, dim=-1)
            return met_mean/constants.met_sol

        sfh_disk_0 = self.lib_ssp.sum_over_age(sfh_disk_0)
        sfh_bulge_0 = self.lib_ssp.sum_over_age(sfh_bulge_0)
        if separate:
            met_disk = mass_weighted_metallicity(self.lib_ssp.met, sfh_disk_0)
            met_bulge = mass_weighted_metallicity(self.lib_ssp.met, sfh_bulge_0)
            return met_disk, met_bulge
        else:
            return mass_weighted_metallicity(self.lib_ssp.met, sfh_disk_0 + sfh_bulge_0)


    def recover_sfh(self, gp_0, sfh_disk, sfh_bulge):
        def recover(sfh, l_norm):
            return self.lib_ssp.reshape_sfh(sfh)/self.lib_ssp.norm*l_norm[:, None, None]

        l_norm = self.helper.get_item(gp_0, 'l_norm')
        b_to_t = self.helper.get_item(gp_0, 'b_to_t')

        sfh_disk_0 = recover(sfh_disk, l_norm*(1 - b_to_t))
        sfh_bulge_0 = recover(sfh_bulge, l_norm*b_to_t)

        return sfh_disk_0, sfh_bulge_0

