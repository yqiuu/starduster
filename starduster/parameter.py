import torch
from torch import nn


class ParameterSet(nn.Module):
    def __init__(self, params_default, free_inds):
        super().__init__()
        self.register_buffer('params_default', params_default)
        self.free_inds = free_inds
        if type(free_inds) is slice:
            self.all_fixed = False
        else:
            self.all_fixed = len(free_inds) == 0


    def forward(self, params):
        return self.derive_full_params(self.set_fixed_params(params))


    def set_fixed_params(self, params):
        if self.all_fixed:
            return self.params_default.title((params, 1))
        else:
            params_out = self.params_default.tile((params.size(0), 1))
            params_out[:, self.free_inds] = params
            return params_out


    def derive_full_params(self, params):
        return params


    @property
    def input_size(self):
        return self.params_default.size(0)


class GalaxyParameter:
    def __init__(self, fixed_gp=None):
        if fixed_gp is None:
            self.fixed_gp = {}
        else:
            self.fixed_gp = fixed_gp


    def init(self, helper):
        params_default = torch.zeros(len(helper.header))
        free_inds = []
        for i_k, key in enumerate(helper.header):
            if key in self.fixed_gp:
                params_default[i_k] = self.fixed_gp[key]
            else:
                free_inds.append(i_k)
        return ParameterSet(params_default, free_inds)


class DiscreteSFH:
    def __init__(self, fixed_sfh=None):
        self.fixed_sfh = fixed_sfh


    def init(self, lib_ssp):
        if self.fixed_sfh is None:
            params_default = torch.zeros(lib_ssp.n_ssp, dtype=torch.float32)
            free_inds = slice(0, lib_ssp.n_ssp)
        else:
            assert self.fixed_sfh.sum() == 1., "Star foramtion history should be normalised to one."
            params_default = torch.ravel(self.fixed_sfh)
            free_inds = ()
        return ParameterSet(params_default, free_inds)

