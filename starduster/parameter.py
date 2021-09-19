import numpy as np
import torch
from torch import nn


class ParameterSet(nn.Module):
    def __init__(self, bounds, params_default, free_inds):
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
            self.bounds = np.asarray(bounds)[self.free_inds]


    def forward(self, params):
        return self.derive_full_params(self.set_fixed_params(params))


    def set_fixed_params(self, params):
        params_out = self.params_default.tile((params.size(0), 1))
        params_out[:, self.free_inds] = params
        return params_out


    def derive_full_params(self, params):
        return params


class InverseDistanceWeightedSFH(ParameterSet):
    def __init__(self, lib_ssp):
        bounds = [(0, 1)]*4
        super().__init__(bounds, torch.zeros(4), slice(0, 4))
        met_sol = 0.019
        self.register_buffer('log_met', torch.log10(lib_ssp.met/met_sol)[:, None])
        self.register_buffer('log_tau', torch.log10(lib_ssp.tau)[:, None])


    def derive_full_params(self, params):
        eps = 1e-6
        log_met, s_met, log_tau, s_tau = params.T
        inv = 1./(torch.abs(log_met - self.log_met)**s_met + eps)
        w_met = (inv/inv.sum(dim=0)).T
        inv = 1./(torch.abs(log_tau - self.log_tau)**s_tau + eps)
        w_tau = (inv/inv.sum(dim=0)).T
        return w_met[:, :, None]*w_tau[:, None, :]


class GalaxyParameter:
    def __init__(self, fixed_gp=None):
        if fixed_gp is None:
            self.fixed_gp = {}
        else:
            self.fixed_gp = fixed_gp


    def init(self, helper):
        bounds = [(-1., 1.)]*len(helper.header)
        params_default = torch.zeros(len(helper.header))
        free_inds = []
        for i_k, key in enumerate(helper.header):
            if key in self.fixed_gp:
                params_default[i_k] = self.fixed_gp[key]
            else:
                free_inds.append(i_k)
        return ParameterSet(bounds, params_default, free_inds)


class DiscreteSFH:
    def __init__(self, fixed_sfh=None):
        self.fixed_sfh = fixed_sfh


    def init(self, lib_ssp):
        bounds = [(0., 1.)]*lib_ssp.n_ssp
        if self.fixed_sfh is None:
            params_default = torch.zeros(lib_ssp.n_ssp, dtype=torch.float32)
            free_inds = slice(0, lib_ssp.n_ssp)
        else:
            assert self.fixed_sfh.sum() == 1., "Star foramtion history should be normalised to one."
            params_default = torch.ravel(self.fixed_sfh)
            free_inds = ()
        return ParameterSet(bounds, params_default, free_inds)

