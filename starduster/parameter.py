import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def partial_init(cls):
    class Wrapper:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.cls = cls


        def init(self, arg_reserved):
            return self.cls(arg_reserved, *self.args, **self.kwargs)

    Wrapper.__name__ = cls.__name__
    Wrapper.__doc__ = cls.__doc__
    return Wrapper


class ParameterSet(nn.Module):
    def __init__(self, bounds, params_default, free_inds, clip_bounds):
        super().__init__()
        self.register_buffer('params_default', params_default)
        self.free_inds = free_inds
        if type(free_inds) is slice:
            self.input_size = free_inds.stop - free_inds.start
        else:
            self.input_size = len(free_inds)
        if self.input_size == 0:
            self.bounds = np.empty((0, 2), dtype=np.float)
        else:
            bounds = np.asarray(bounds)[self.free_inds]
            lbounds, ubounds = torch.tensor(bounds, dtype=torch.float32).T
            self.register_buffer('lbounds', lbounds)
            self.register_buffer('ubounds', ubounds)
            self.register_buffer('bound_radius', .5*(ubounds - lbounds))
            self.register_buffer('bound_centre', .5*(ubounds + lbounds))
            self.bounds = bounds
        self.clip_bounds = clip_bounds


    def forward(self, params, check_bounds=False):
        if check_bounds:
            if self.input_size == 0:
                is_out = torch.full((params.size(0),), False)
            else:
                is_out = torch.any(params <= self.lbounds, dim=-1) \
                    | torch.any(params >= self.ubounds, dim=-1)

        if self.clip_bounds:
            eps = 1e-6
            params = (params - self.bound_centre)/self.bound_radius
            params = F.hardtanh(params, -1 + eps, 1 - eps)
            params = self.bound_radius*params + self.bound_centre

        params = self.derive_full_params(self.set_fixed_params(params))

        if check_bounds:
            return params, is_out
        else:
            return params


    def set_fixed_params(self, params):
        params_out = self.params_default.tile((params.size(0), 1))
        params_out[:, self.free_inds] = params
        return params_out


    def derive_full_params(self, params):
        return params


@partial_init
class GalaxyParameter(ParameterSet):
    def __init__(self, helper, clip_bounds=True, **fixed_params):
        bounds = [(-1., 1.)]*len(helper.header)
        params_default, free_inds = derive_default_params(helper.header.keys(), fixed_params)
        super().__init__(bounds, params_default, free_inds, clip_bounds)


@partial_init
class DiscreteSFH(ParameterSet):
    def __init__(self, lib_ssp, fixed_sfh=None, clip_bounds=True):
        bounds = [(0., 1.)]*lib_ssp.n_ssp
        if fixed_sfh is None:
            params_default = torch.zeros(lib_ssp.n_ssp, dtype=torch.float32)
            free_inds = slice(0, lib_ssp.n_ssp)
        else:
            assert fixed_sfh.sum() == 1., "Star foramtion history should be normalised to one."
            params_default = torch.ravel(fixed_sfh)
            free_inds = ()
        super().__init__(bounds, params_default, free_inds, clip_bounds)


@partial_init
class DiscreteSFR_InterpolatedMet(ParameterSet):
    # TODO: Add functions to make sure that the SFH is normalised to one.
    # Handle the situation where there are only two SFR bins.
    def __init__(self,
        lib_ssp, sfr_bins=None, uni_met=False, simplex_transform=False,
        clip_bounds=True, **fixed_params
    ):
        self.n_tau_ssp = lib_ssp.n_tau
        if sfr_bins is None:
            n_sfr = lib_ssp.n_tau
        else:
            n_sfr = len(sfr_bins)
        self.sfr_bins = sfr_bins
        self.simplex_transform = simplex_transform

        met_sol = 0.019
        log_met = torch.log10(lib_ssp.met/met_sol)[:, None]
        log_met_min = float(log_met[0])
        log_met_max = float(log_met[-1])
        if uni_met:
            bounds = np.array([(0., 1.)]*n_sfr + [(log_met_min, log_met_max)])
            param_names = [f'sfr_{i_sfr}' for i_sfr in range(n_sfr)] + ['log_met']
        else:
            bounds = np.array([(0., 1.)]*n_sfr + [(log_met_min, log_met_max)]*n_sfr)
            param_names = [f'sfr_{i_sfr}' for i_sfr in range(n_sfr)] \
                + [f'log_met_{i_sfr}' for i_sfr in range(n_sfr)]
        params_default, free_inds = derive_default_params(param_names, fixed_params)
        super().__init__(bounds, params_default, free_inds, clip_bounds)
        self.register_buffer('log_met', log_met)
        self.n_sfr = n_sfr


    def derive_full_params(self, params):
        def simplex_transform(x):
            x = -torch.log(1 - x)
            return x/x.sum(dim=1, keepdim=True)

        sfr = params[:, :self.n_sfr]
        log_met = params[:, self.n_sfr:]
        if self.simplex_transform:
            sfr = simplex_transform(sfr)
        sfh = sfr[:, None, :]*self.derive_idw_met(log_met)
        if self.n_sfr != self.n_tau_ssp:
            sfh_full = torch.zeros([sfh.size(0), sfh.size(1), self.n_tau_ssp],
                dtype=sfh.dtype, layout=sfh.layout, device=sfh.device)
            for i_b, (idx_b, idx_e) in enumerate(self.sfr_bins):
                sfh_full[:, :, idx_b:idx_e] = sfh[:, :, i_b, None]/(idx_e - idx_b)
            sfh = sfh_full
        sfh = torch.flatten(sfh, start_dim=1)
        return sfh


    def derive_idw_met(self, log_met):
        eps = 1e-6
        inv = 1./((log_met[:, None, :] - self.log_met)**2 + eps)
        return inv/inv.sum(dim=1)[:, None, :]


@partial_init
class InverseDistanceWeightedSFH(ParameterSet):
    def __init__(self, lib_ssp, clip_bounds=True, **fixed_params):
        met_sol = 0.019
        log_met = torch.log10(lib_ssp.met/met_sol)[:, None]
        log_tau = torch.log10(lib_ssp.tau)[:, None]
        bounds = np.asarray([
            (float(log_met[0]), float(log_met[-1])), (-2., 1.),
            (float(log_tau[0]), float(log_tau[-1])), (-2., 1.)
        ])
        param_names = ['log_met', 's_met', 'log_tau', 's_tau']
        params_default, free_inds = derive_default_params(param_names, fixed_params)
        super().__init__(bounds, params_default, free_inds, clip_bounds)
        self.register_buffer('log_met', log_met)
        self.register_buffer('log_tau', log_tau)


    def derive_full_params(self, params):
        eps = 1e-6
        log_met, s_met, log_tau, s_tau = params.T
        inv = 1./(torch.abs(log_met - self.log_met)**(10**s_met) + eps)
        w_met = (inv/inv.sum(dim=0)).T
        inv = 1./(torch.abs(log_tau - self.log_tau)**(10**s_tau) + eps)
        w_tau = (inv/inv.sum(dim=0)).T
        return torch.flatten(w_met[:, :, None]*w_tau[:, None, :], start_dim=1)



def derive_default_params(param_names, fixed_params):
    params_default = torch.zeros(len(param_names))
    free_inds = []
    for i_k, key in enumerate(param_names):
        if key in fixed_params:
            params_default[i_k] = fixed_params[key]
        else:
            free_inds.append(i_k)
    return params_default, free_inds

