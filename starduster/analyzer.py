from .selector import sample_from_selector

import torch


class Analyzer:
    def __init__(self, sed_model):
        self.sed_model = sed_model
        self.helper = sed_model.helper
        self.lib_ssp = sed_model.lib_ssp


    def sample(self, n_samp):
        adapter = self.sed_model.adapter
        lb, ub = torch.tensor(self.sed_model.adapter.bounds, dtype=torch.float32).T
        sampler = lambda n_samp: (ub - lb)*torch.rand([n_samp, len(lb)]) + lb
        adapter.selector_disk.cpu()
        adapter.selector_bulge.cpu()
        samps = sample_from_selector(n_samp, adapter.selector_disk, adapter.selector_bulge, sampler)
        adapter.to(adapter.device)
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
        # TODO: put units in somewhere more formal
        units = {
            'theta': '',
            'r_disk': 'kpc',
            'r_bulge': 'kpc',
            'r_dust': 'kpc',
            'l_norm': 'L_sol',
            'b_to_t': '',
            'm_dust': 'M_sol',
            'm_disk': 'M_sol',
            'm_bulge': 'M_sol',
            'm_star': 'M_sol',
            'sfr_10': 'M_sol/yr',
            'sfr_100': 'M_sol/yr'
        }
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
                if key == 'theta':
                    print("{}: {:.1f} {}".format(key, val[0], units[key]))
                elif key == 'b_to_t':
                    print("{}: {:.2f} {}".format(key, val[0], units[key]))
                else:
                    if log_scale:
                        print("{}: {:.2f} {}".format(key, val[0], units[key]))
                    else:
                        print("{}: {:.2e} {}".format(key, val[0], units[key]))

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

