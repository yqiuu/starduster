import numpy as np
import torch
from torch import nn


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


@partial_init
class GalaxyParameter(ParameterSet):
    def __init__(self, helper, fixed_gp=None):
        bounds = [(-1., 1.)]*len(helper.header)
        params_default, free_inds = derive_default_params(helper.header.keys(), fixed_gp)
        super().__init__(bounds, params_default, free_inds)


@partial_init
class DiscreteSFH(ParameterSet):
    def __init__(self, lib_ssp, fixed_sfh=None):
        bounds = [(0., 1.)]*lib_ssp.n_ssp
        if fixed_sfh is None:
            params_default = torch.zeros(lib_ssp.n_ssp, dtype=torch.float32)
            free_inds = slice(0, lib_ssp.n_ssp)
        else:
            assert fixed_sfh.sum() == 1., "Star foramtion history should be normalised to one."
            params_default = torch.ravel(fixed_sfh)
            free_inds = ()
        super().__init__(bounds, params_default, free_inds)


@partial_init
class InverseDistanceWeightedSFH(ParameterSet):
    def __init__(self, lib_ssp, fixed_sfh=None):
        met_sol = 0.019
        log_met = torch.log10(lib_ssp.met/met_sol)[:, None]
        log_tau = torch.log10(lib_ssp.tau)[:, None]
        bounds = np.asarray([
            (float(log_met[0]), float(log_met[-1])), (-2., 1.),
            (float(log_tau[0]), float(log_tau[-1])), (-2., 1.)
        ])
        param_names = ['log_met', 's_met', 'log_tau', 's_tau']
        params_default, free_inds = derive_default_params(param_names, fixed_sfh)
        super().__init__(bounds, params_default, free_inds)
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
    if fixed_params is None:
        fixed_params = {}
    params_default = torch.zeros(len(param_names))
    free_inds = []
    for i_k, key in enumerate(param_names):
        if key in fixed_params:
            params_default[i_k] = fixed_params[key]
        else:
            free_inds.append(i_k)
    return params_default, free_inds
           
