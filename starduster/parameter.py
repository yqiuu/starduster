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

