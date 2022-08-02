from .utils import constants, simple_trapz, simplex_transform, compute_interp_weights

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Parametrization(nn.Module):
    """Base class for a parametrization of a SED model.

    The initialization of this class plays a trick to prevent the user from
    passing extra variables. When writing a subclass, the developer should
    following convention:

    def __init__(self, A, B, C):
        super().__init__(A, B, C)

    def _init(self, helper, lib_ssp, A, B, C):
        # Write the code for initialization here.

    Call ``enable`` to make this class work, which executes ``_init``
    internally.
    """
    def __init__(self, *args):
        super().__init__()
        self._args = args


    def enable(self, helper, lib_ssp):
        """Eable the class.

        Parameters
        ----------
        helper : Helper
            Helper of input parameters.
        lib_ssp : SSPLibrary
            A simple stellar population library.
        """
        param_names, fixed_params, bounds_default, bounds, boundary \
            = self._init(helper, lib_ssp, *self._args)
        params_default, free_inds = self._derive_default_params(param_names, fixed_params)
        self._update_bounds(param_names, bounds_default, bounds)
        self.register_buffer('params_default', params_default)
        self.free_inds = free_inds
        if type(free_inds) is slice:
            self.input_size = free_inds.stop - free_inds.start
        else:
            self.input_size = len(free_inds)
        if self.input_size == 0:
            self.param_names = np.empty(0, dtype=np.str)
            self.bounds = np.empty((0, 2), dtype=np.float)
        else:
            self.param_names = np.asarray(list(param_names))[free_inds]
            bounds = np.asarray(bounds_default)[free_inds]
            lbounds, ubounds = torch.tensor(bounds, dtype=torch.float32).T
            self.register_buffer('lbounds', lbounds)
            self.register_buffer('ubounds', ubounds)
            self.register_buffer('bound_radius', .5*(ubounds - lbounds))
            self.register_buffer('bound_centre', .5*(ubounds + lbounds))
            self.bounds = bounds
        self.boundary = boundary


    def _init(self, helper, lib_ssp, *args):
        """The initialization body.

        Parameters
        ----------
        helper : Helper
            Helper of input parameters.
        lib_ssp : SSPLibrary
            A simple stellar population library.
        args : tuple
            Any arguments passed to ``__init__``.

        Returns
        -------
        param_names : list
            List of parameter names.
        fixed_params : dict
            A dictionary to specify fixed parameters. Use the name of the
            parameter as the key.
        bounds_default : array
            An array of (min, max) to specify the default bounds of the
            parameters.
        bounds : array
            An array of (min, max) to specify the working bounds of the
            parameters.
        boundary : str {'none', 'clipping', 'reflecting', 'absorbing'}
            - 'none': Do not apply any transform.
            - 'clipping': Clip the parameters when they are beyond the bounds.
            - 'reflecting': Apply reflecting boundary condition to the
              parameters.
            - 'absorbing': Uniformly sample parameters when they are beyond the
              bounds.
        """
        # return param_names, fixed_params, bounds_default, bounds, boundary
        pass


    def _derive_default_params(self, param_names, fixed_params):
        params_default = torch.zeros(len(param_names))
        free_inds = []
        for i_k, key in enumerate(param_names):
            if key in fixed_params:
                params_default[i_k] = fixed_params[key]
            else:
                free_inds.append(i_k)
        return params_default, free_inds


    def _update_bounds(self, param_names, bounds_default, bounds):
        if bounds is None:
            return

        for i_k, key in enumerate(param_names):
            if key in bounds:
                lb, ub = bounds_default[i_k]
                lb_new, ub_new = bounds[key]
                if lb_new >= lb and ub_new <= ub:
                    bounds_default[i_k] = (lb_new, ub_new)
                else:
                    msg = "The bounds of {} should be within [{:.2f}, {:.2f}]".format(key, lb, ub)
                    raise ValueError(msg)


    def forward(self, params):
        params = self.boundary_transform(params)
        params = self._set_fixed_params(params)
        params = self.derive_full_params(params)
        return params


    def derive_full_params(self, params):
        """Implementation of the parametrization.

        Should be overridden by all subclasses.
        """
        return params


    def _set_fixed_params(self, params):
        params_out = self.params_default.tile((params.size(0), 1))
        params_out[:, self.free_inds] = params
        return params_out


    def check_bounds(self, params):
        if self.input_size == 0:
            return torch.full((params.size(0),), False)
        else:
            return torch.any(params <= self.lbounds, dim=-1) \
                | torch.any(params >= self.ubounds, dim=-1)


    def boundary_transform(self, params):
        if self.input_size > 0:
            if self.boundary == 'none':
                pass
            elif self.boundary == 'clipping':
                params = self._clipping_boundary(params)
            elif self.boundary == 'reflecting':
                params = self._reflecting_boundary(params)
            elif self.boundary == 'absorbing':
                params = self._absorbing_boundary(params)
            else:
                raise ValueError(f"Unknown boundary: {self.boundary}.")
        return params


    def _clipping_boundary(self, params):
        eps = 1e-6
        params = (params - self.bound_centre)/self.bound_radius
        params = F.hardtanh(params, -1 + eps, 1 - eps)
        params = self.bound_radius*params + self.bound_centre
        return params


    def _reflecting_boundary(self, params):
        cond_left = params < self.lbounds
        cond_right = params >= self.ubounds

        params = (params - self.lbounds)/(self.ubounds - self.lbounds)
        params =  (1 - torch.exp(params))*cond_left \
            + params*(~cond_left & ~cond_right) \
            + torch.exp(1 - params)*cond_right
        params = self.lbounds + (self.ubounds - self.lbounds)*params
        return params


    def _absorbing_boundary(self, params):
        cond = (params >= self.lbounds) & (params < self.ubounds)
        params_rand = self.lbounds + (self.ubounds - self.lbounds)*torch.rand_like(params)
        return params*cond + params_rand*~cond


class GalaxyParameter(Parametrization):
    """A class to configure galaxy parameters.

    Parameters
    ----------
    sed_model : MultiwavelengthSED
        The target SED model.
    bounds : array
        An array of (min, max) to specify the working bounds of the parameters.
    boundary : bool
        If true, when an input value is beyond a bound, set it to be the same
        with the bound.
    fixed_params : dict
        A dictionary to specify fixed parameters. Use the name of the parameter
        as the key.
    """
    def __init__(self, bounds=None, boundary='clipping', **fixed_params):
        super().__init__(bounds, boundary, fixed_params)


    def _init(self, helper, lib_ssp, bounds, boundary, fixed_params):
        param_names = helper.header.keys()
        bounds_default = [(-1., 1.)]*len(helper.header)
        # Transform the parameters
        if bounds is not None:
            bounds = bounds.copy()
            for key, (lb, ub) in bounds.items():
                lb = helper.transform(lb, key)
                ub = helper.transform(ub, key)
                if key == 'theta':
                    bounds[key] = (ub, lb)
                else:
                    bounds[key] = (lb, ub)
        # Transform the parameters
        fixed_params = fixed_params.copy()
        for key, val in fixed_params.items():
            fixed_params[key] = helper.transform(val, key)
        return param_names, fixed_params, bounds_default, bounds, boundary


class VanillaGrid(Parametrization):
    """Default star formation history grid.

    Parameters
    ----------
    sed_model : MultiwavelengthSED
        The target SED model.
    simplex_transform : bool
        If true, apply a transform to the input which maps a unit hypercube
        into a simplex.
    bounds : array
        An array of (min, max) to specify the working bounds of the parameters.
    boundary : bool
        If true, when an input value is beyond a bound, set it to be the same
        with the bound.
    fixed_params : dict
        A dictionary to specify fixed parameters. Use the name of the parameter
        as the key.
    """
    def __init__(self, simplex_transform=False, bounds=None, boundary='clipping', **fixed_params):
        super().__init__(simplex_transform, bounds, boundary, fixed_params)


    def _init(self, helper, lib_ssp, simplex_transform, bounds, boundary, fixed_params):
        param_names = [f'sfr_{i_sfr}' for i_sfr in range(lib_ssp.n_ssp)]
        bounds_default = [(0., 1.)]*lib_ssp.n_ssp

        self.simplex_transform = simplex_transform

        return param_names, fixed_params, bounds_default, bounds, boundary


    def derive_full_params(self, params):
        if self.simplex_transform:
            return simplex_transform(params)
        else:
            return params


class CompositeGrid(Parametrization):
    """A grid that combines a star formation history and a metallicity history.

    Parameters
    ----------
    sfh_model : SFHComponent
        Parametrization of the star formation history.
    mh_model : SFHComponent
        Parametrization of the metallicity history.
    bounds : array
        An array of (min, max) to specify the working bounds of the parameters.
    boundary : bool
        If true, when an input value is beyond a bound, set it to be the same
        with the bound.
    fixed_params : dict
        A dictionary to specify fixed parameters. Use the name of the parameter
        as the key.
    """
    def __init__(self, sfh_model, mh_model, bounds=None, boundary='clipping', **fixed_params):
        super().__init__(sfh_model, mh_model, bounds, boundary, fixed_params)


    def _init(self, helper, lib_ssp, sfh_model, mh_model, bounds, boundary, fixed_params):
        param_names_sfh, bounds_default_sfh = sfh_model.enable(lib_ssp)
        param_names_mh, bounds_default_mh = mh_model.enable(lib_ssp)
        param_names = param_names_sfh + param_names_mh
        bounds_default = np.vstack([bounds_default_sfh, bounds_default_mh])

        self.split_size = (len(param_names_sfh), len(param_names_mh))
        self.sfh_model = sfh_model
        self.mh_model = mh_model
        self.need_light_norm = getattr(sfh_model, 'need_light_norm', False)
        self.register_buffer('light_norm', torch.ravel(lib_ssp.norm))

        return param_names, fixed_params, bounds_default, bounds, boundary


    def derive_full_params(self, params):
        params_sfh, params_mh = torch.split(params, self.split_size, dim=1)
        sfh = self.sfh_model(params_sfh)
        mh = self.mh_model(params_mh, sfh)
        sfh_grid = torch.flatten(sfh[:, None, :]*mh, start_dim=1)
        if self.need_light_norm:
            sfh_grid = sfh_grid*self.light_norm
            sfh_grid = sfh_grid/sfh_grid.sum(dim=-1, keepdim=True)
        return sfh_grid


class SFHComponent(nn.Module):
    """Base class for a star formation history component.

    The initialization of this class plays a trick to prevent the user from
    passing extra variables. When writing a subclass, the developer should
    following convention:

    def __init__(self, A, B, C):
        super().__init__(A, B, C)

    def _init(self, lib_ssp, A, B, C):
        # Write the code for initialization here.

    Call ``enable`` to make this class work, which executes ``_init``
    internally.
    """
    def __init__(self, *args):
        super().__init__()
        self._args = args


    def enable(self, lib_ssp):
        """Eable the class.

        Parameters
        ----------
        lib_ssp : SSPLibrary
            A simple stellar population library.
        """
        param_names, bounds_default = self._init(lib_ssp, *self._args)
        return param_names, bounds_default


    def _init(self, lib_ssp, *args):
        """The initialization body.

        Parameters
        ----------
        lib_ssp : SSPLibrary
            A simple stellar population library.
        args : tuple
            Any arguments passed to ``__init__``.

        Returns
        -------
        param_names : list
            List of parameter names.
        bounds_default : array
            An array of (min, max) to specify the default bounds of the
            parameters.
        """
        # return param_names, bounds_default
        pass


class DiscreteSFH(SFHComponent):
    """Discrete star formation history.

    Parameters
    ----------
    sfh_bins : list
        Each element should be a doublet specifying the start and end bin
        indices.
    simplex_transform : bool
        If true, apply a transform to the input which maps a unit hypercube
        into a simplex.
    """
    def __init__(self, sfh_bins=None, simplex_transform=False):
        super().__init__(sfh_bins, simplex_transform)


    def _init(self, lib_ssp, sfh_bins, simplex_transform):
        n_tau = lib_ssp.n_tau
        if sfh_bins is None:
            n_sfh = n_tau
        else:
            self._check_sfh_bins(sfh_bins, n_tau)
            n_sfh = len(sfh_bins)

        if n_sfh == 1:
            param_names = []
            bounds_default = np.zeros((0, 2))
        elif n_sfh == 2:
            # There is one degree of freedom in this case due to the
            # normalization condition
            param_names = ['c_0']
            bounds_default = np.array([0., 1.])
        else:
            param_names = [f'c_{i_sfh}' for i_sfh in range(n_sfh)]
            bounds_default = np.tile((0., 1.), (n_sfh, 1))

        self.simplex_transform = simplex_transform
        self.sfh_bins = sfh_bins
        self.n_tau_ssp = n_tau
        self.n_sfh = n_sfh
        return param_names, bounds_default


    def _check_sfh_bins(self, sfh_bins, n_tau):
        idx_prev = -1
        for idx_b, idx_e in sfh_bins:
            assert idx_b >= 0 and idx_e >= 0 and idx_b <= n_tau and idx_e <= n_tau, \
                f"Indices of 'sfh_bins' must be within [0, {n_tau}]."
            assert idx_b < idx_e, f"Invalid 'sfh_bins': [{idx_b}, {idx_e}]."
            assert idx_b >= idx_prev, "'sfh_bins' must not have overlaps."
            idx_prev = idx_e


    def forward(self, params):
        # (N, *) -> (N, N_age)
        if self.simplex_transform and self.n_sfh > 2:
            params = simplex_transform(params)
        if self.sfh_bins is None:
            sfh = params
        else:
            sfh = torch.zeros([params.size(0), self.n_tau_ssp],
                dtype=params.dtype, layout=params.layout, device=params.device)
            if self.n_sfh == 1:
                idx_b, idx_e = self.sfh_bins[0]
                sfh[:, idx_b:idx_e] = 1./(idx_e - idx_b)
            elif self.n_sfh == 2:
                (idx_b_0, idx_e_0), (idx_b_1, idx_e_1) = self.sfh_bins
                sfh[:, idx_b_0:idx_e_0] = params/(idx_e_0 - idx_b_0)
                sfh[:, idx_b_1:idx_e_1] = (1 - params)/(idx_e_1 - idx_b_1)
            else:
                for i_bin, (idx_b, idx_e) in enumerate(self.sfh_bins):
                    sfh[:, idx_b:idx_e] = params[:, i_bin, None]/(idx_e - idx_b)
        return sfh


class InterpolatedSFH(SFHComponent):
    """Linearly interpolate the input stellar age into the grid."""
    def _init(self, lib_ssp, *args):
        log_tau = torch.log10(lib_ssp.tau)
        self.register_buffer('log_tau', log_tau)
        param_names = ['log_tau']
        bounds_default = np.array([[log_tau[0].item(),log_tau[-1].item()]])
        return param_names, bounds_default


    def forward(self, params):
        # (N, 1) -> (N, N_age)
        params = params.contiguous()
        return torch.squeeze(compute_interp_weights(params, self.log_tau), dim=1)


class AnalyticSFH(SFHComponent):
    """Base class for an analytic SFH model.

    Parameters
    ----------
    n_sub : int
        Number of points for the integration.
    """
    def __init__(self, n_sub=16):
        self.n_sub = n_sub
        super().__init__()


    def enable(self, lib_ssp):
        self.need_light_norm = True
        self.register_buffer('t_age', self._create_t_age(lib_ssp.tau_edges))
        return super().enable(lib_ssp)


    def _create_t_age(self, tau_edges):
        n_sub = self.n_sub
        if n_sub == 1:
            return tau_edges

        n_age = tau_edges.size(0)
        log_tau_edges = np.log10(tau_edges)
        t_age = torch.zeros(n_age + (n_sub - 1)*(n_age - 1))
        for i_age in range(n_age - 1):
            lower = log_tau_edges[i_age]
            upper = log_tau_edges[i_age + 1]
            t_age[i_age*n_sub : (i_age + 1)*n_sub] = torch.logspace(lower, upper, n_sub + 1)[:-1]
        t_age[-1] = tau_edges[-1]
        return t_age


    def forward(self, params):
        params, t_trunc = self._preprocess_params(params)
        if t_trunc is None:
            t_age = self.t_age.unsqueeze(0)
        else:
            t_age = t_trunc*F.hardtanh(self.t_age/t_trunc)
        sfh = simple_trapz(self.derive_sfh(t_age, params, t_trunc), t_age)
        if self.n_sub == 1:
            return sfh
        else:
            return torch.sum(sfh.view(sfh.size(0), -1, self.n_sub), dim=-1)


    def derive_sfh(self, t_age, *args):
        pass


    def _preprocess_params(self, params):
        return params, None


class ExponentialSFH(AnalyticSFH):
    r"""Exponentially declining SFH.

    The model as a function of stellar age is:

    .. math::
        \text{SFH}(t) \propto \begin{cases}
            e^{t/\tau} & \text{ if } t < t_\text{trunc} \\
            0 & \text{ otherwise } \\
        \end{cases}

    Parameters
    ----------
    n_sub : int
        Number of points for the integration.
    """
    def _init(self, lib_ssp, *args):
        param_names = ['log10_tau', 'log10_t0']
        log_t0_min = math.log10(lib_ssp.tau_edges[0])
        log_t0_max = math.log10(lib_ssp.tau_edges[-1])
        bounds_default = np.array([(log_t0_max - 1.5, log_t0_max + 1.5), (log_t0_min, log_t0_max)])
        return param_names, bounds_default


    def derive_sfh(self, t_age, tau, t_trunc):
        return torch.exp(t_age/tau)


    def _preprocess_params(self, params):
        return torch.hsplit(10**params, (1,))


class DelayedExponentialSFH(AnalyticSFH):
    r"""Delayed exponentially declining SFH.

    The model as a function of stellar age is:

    .. math::
        \text{SFH}(t) \propto \begin{cases}
            (t_\text{trunc} - t) e^{t/\tau} & \text{ if } t < t_\text{trunc} \\
            0 & \text{ otherwise } \\
        \end{cases}

    Parameters
    ----------
    n_sub : int
        Number of points for the integration.
    """
    def _init(self, lib_ssp, *args):
        param_names = ['log10_tau', 'log10_t0']
        log_t0_min = math.log10(lib_ssp.tau_edges[1])
        log_t0_max = math.log10(lib_ssp.tau_edges[-1])
        bounds_default = np.array([(log_t0_max - 1.5, log_t0_max + 1.5), (log_t0_min, log_t0_max)])
        return param_names, bounds_default


    def derive_sfh(self, t_age, tau, t_trunc):
        return (t_trunc - t_age)*torch.exp(t_age/tau)


    def _preprocess_params(self, params):
        return torch.hsplit(10**params, (1,))


class InterpolatedMH(SFHComponent):
    """Linearly interpolate the input metallicity into the grid."""
    def _init(self, lib_ssp, *args):
        n_met = len(lib_ssp.met)
        log_met = torch.log10(lib_ssp.met/constants.met_sol)
        self.register_buffer('log_met', log_met)
        param_names = ['log_met']
        bounds_default = np.array([[log_met[0].item(), log_met[-1].item()]])
        return param_names, bounds_default


    def forward(self, params, sfh):
        # params (N, 1) -> (N, N_met, N_age)
        params = params.contiguous()
        return torch.swapaxes(compute_interp_weights(params, self.log_met), 1, 2)


class ClosedBoxMH(SFHComponent):
    """Metallicity history that is proportional to the cumulative SFH.

    This model can be found in Robotham et al. 2020, which approximates the
    closed box metallicity evolution.
    """
    def _init(self, lib_ssp, *args):
        met = lib_ssp.met/constants.met_sol
        self.register_buffer('met', met)
        self.register_buffer('met_min', met[0])
        param_names = ['log10_met_max', 'log10_frac']
        bounds_default = np.array([
            [math.log10(met[0].item()), math.log10(met[-1].item())], [-3, 0]
        ])
        return param_names, bounds_default


    def forward(self, params, sfh):
        params = 10**params
        met_max = params[:, :1]
        met_min = met_max - (met_max - self.met_min)*(1 - params[:, 1:])
        csfh = torch.cumsum(sfh, dim=1)
        csfh = torch.flip(csfh/csfh[:, -1:], dims=[1])
        met = met_min + (met_max - met_min)*csfh
        met = torch.swapaxes(compute_interp_weights(met, self.met), 1, 2)
        return met

