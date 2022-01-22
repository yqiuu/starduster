from .utils import constants, simple_trapz

import math
from bisect import bisect_left

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
        param_names, fixed_params, bounds_default, bounds, clip_bounds \
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
        self.clip_bounds = clip_bounds


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
        clip_bounds : bool
            If true, when an input value is beyond a bound, set it to be the
            same with the bound.
        """
        # return param_names, fixed_params, bounds_default, bounds, clip_bounds
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
        params = self._clip_bounds(params)
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


    def _clip_bounds(self, params):
        if self.clip_bounds and self.input_size > 0:
            eps = 1e-6
            params = (params - self.bound_centre)/self.bound_radius
            params = F.hardtanh(params, -1 + eps, 1 - eps)
            params = self.bound_radius*params + self.bound_centre
        return params


    def check_bounds(self, params):
        if self.input_size == 0:
            return torch.full((params.size(0),), False)
        else:
            return torch.any(params <= self.lbounds, dim=-1) \
                | torch.any(params >= self.ubounds, dim=-1)


class GalaxyParameter(Parametrization):
    """A class to configure galaxy parameters.

    Parameters
    ----------
    sed_model : MultiwavelengthSED
        The target SED model.
    bounds : array
        An array of (min, max) to specify the working bounds of the parameters.
    clip_bounds : bool
        If true, when an input value is beyond a bound, set it to be the same
        with the bound.
    fixed_params : dict
        A dictionary to specify fixed parameters. Use the name of the parameter
        as the key.
    """
    def __init__(self, bounds=None, clip_bounds=True, **fixed_params):
        super().__init__(bounds, clip_bounds, fixed_params)


    def _init(self, helper, lib_ssp, bounds, clip_bounds, fixed_params):
        param_names = helper.header.keys()
        bounds_default = [(-1., 1.)]*len(helper.header)
        # Transform the parameters
        if bounds is not None:
            for key, (lb, ub) in bounds.items():
                lb = helper.transform(lb, key)
                ub = helper.transform(ub, key)
                if key == 'theta':
                    bounds[key] = (ub, lb)
                else:
                    bounds[key] = (lb, ub)
        # Transform the parameters
        for key, val in fixed_params.items():
            fixed_params[key] = helper.transform(val, key)
        return param_names, fixed_params, bounds_default, bounds, clip_bounds


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
    clip_bounds : bool
        If true, when an input value is beyond a bound, set it to be the same
        with the bound.
    fixed_params : dict
        A dictionary to specify fixed parameters. Use the name of the parameter
        as the key.
    """
    def __init__(self, simplex_transform=False, bounds=None, clip_bounds=True, **fixed_params):
        super().__init__(simplex_transform, bounds, clip_bounds, fixed_params)


    def _init(self, helper, lib_ssp, simplex_transform, bounds, clip_bounds, fixed_params):
        param_names = [f'sfr_{i_sfr}' for i_sfr in range(lib_ssp.n_ssp)]
        bounds_default = [(0., 1.)]*lib_ssp.n_ssp

        self.simplex_transform = simplex_transform

        return param_names, fixed_params, bounds_default, bounds, clip_bounds


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
    clip_bounds : bool
        If true, when an input value is beyond a bound, set it to be the same
        with the bound.
    fixed_params : dict
        A dictionary to specify fixed parameters. Use the name of the parameter
        as the key.
    """
    def __init__(self, sfh_model, mh_model, bounds=None, clip_bounds=True, **fixed_params):
        super().__init__(sfh_model, mh_model, bounds, clip_bounds, fixed_params)


    def _init(self, helper, lib_ssp, sfh_model, mh_model, bounds, clip_bounds, fixed_params):
        param_names_sfh, bounds_default_sfh = sfh_model.enable(lib_ssp)
        param_names_mh, bounds_default_mh = mh_model.enable(lib_ssp)
        param_names = param_names_sfh + param_names_mh
        bounds_default = np.vstack([bounds_default_sfh, bounds_default_mh])

        self.split_size = (len(param_names_sfh), len(param_names_mh))
        self.sfh_model = sfh_model
        self.mh_model = mh_model
        self.need_light_norm = getattr(sfh_model, 'need_light_norm', False)
        self.register_buffer('light_norm', torch.ravel(lib_ssp.norm))

        return param_names, fixed_params, bounds_default, bounds, clip_bounds


    def derive_full_params(self, params):
        params_sfh, params_mh = torch.split(params, self.split_size, dim=1)
        sfh = self.sfh_model.derive(params_sfh)
        mh = self.mh_model.derive(params_mh, sfh)
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


    def derive(self, *args):
        """Implementation of the parametrization.

        Should be overridden by all subclasses.
        """
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


    def derive(self, params):
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


    def derive(self, params):
        # (N, 1) -> (N, N_age)
        params = params.contiguous()
        return torch.squeeze(compute_interp_weights(params, self.log_tau), dim=1)


class AnalyticSFH(SFHComponent):
    def __init__(self, n_sub=4):
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


    def derive(self, params):
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
    def _init(self, lib_ssp, *args):
        param_names = ['log10_tau', 'log10_t0']
        log_t0_min = math.log10(lib_ssp.tau_edges[0])
        log_t0_max = math.log10(lib_ssp.tau_edges[-1])
        bounds_default = np.array([(log_t0_max - 1, log_t0_max + 2), (log_t0_min, log_t0_max)])
        return param_names, bounds_default


    def derive_sfh(self, t_age, tau, t_trunc):
        return torch.exp(t_age/tau)


    def _preprocess_params(self, params):
        return torch.hsplit(10**params, (1,))


class DelayedExponentialSFH(AnalyticSFH):
    def _init(self, lib_ssp, *args):
        param_names = ['log10_tau', 'log10_t0']
        log_t0_min = math.log10(lib_ssp.tau_edges[1])
        log_t0_max = math.log10(lib_ssp.tau_edges[-1])
        bounds_default = np.array([(log_t0_max - 1, log_t0_max + 2), (log_t0_min, log_t0_max)])
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


    def derive(self, params, sfh):
        # params (N, 1) -> (N, N_met, N_age)
        params = params.contiguous()
        return torch.swapaxes(compute_interp_weights(params, self.log_met), 1, 2)


class SemiAnalyticConventer:
    """A class to convert properties from a semi-analytic model into parameters
    that can be accepted by the given SED model.

    Parameters
    ----------
    sed_model : MultiwavelengthSED
        The target SED model.
    age_bins : array [yr]
        Stellar age bins.
    """
    def __init__(self, sed_model, age_bins):
        self.helper = sed_model.helper
        self.lib_ssp = sed_model.lib_ssp
        self._tau_matrix = self._create_tau_matrix(age_bins)


    def __call__(self,
        theta, m_dust, r_dust, r_disk, r_bulge,
        sfh_mass_disk, sfh_metal_mass_disk, sfh_mass_bulge, sfh_metal_mass_bulge
    ):
        """Convert the input properties into parameters that can be accepted by
        the given SED model.

        Parameters
        ----------
        theta : array [deg]
            Inlincation angel.
        m_dust : array [M_sol]
            Dust mass.
        r_dust : array [kpc]
            Dust disk radius.
        r_disk : array [kpc]
            Stellar disk radius.
        r_bulge : array [kpc]
            Stellar bulge radius.
        sfh_mass_disk : array [M_sol]
            Gross stellar mass of the stellar disk in each age bin.
        sfh_metal_mass_disk : array [M_sol]
            Metal mass of the stellar disk in each age bin.
        sfh_mass_bulge : array [M_sol]
            Gross stellar mass of the stellar bulge in each age bin.
        sfh_metal_mass_bulge : array [M_sol]
            Metal mass of the stellar bulge in each age bin.

        Returns
        -------
        gp : tensor
            Galaxy parameters.
        sfh_disk : tensor
            Star formation history of the disk component.
        sfh_bulge : tensor
            Star formation history of the bulge component.
        """
        den_dust = m_dust/(2*np.pi*r_dust*r_dust)
        r_dust_to_rd = r_dust/r_disk
        sfh_disk, l_norm_disk = self._derive_sfh(sfh_mass_disk, sfh_metal_mass_disk)
        sfh_bulge, l_norm_bulge = self._derive_sfh(sfh_mass_bulge, sfh_metal_mass_bulge)
        l_norm = l_norm_disk + l_norm_bulge
        b_to_t = l_norm_bulge/l_norm
        gp_0 = np.vstack([theta, den_dust, r_dust_to_rd, r_disk, r_bulge, l_norm, b_to_t]).T
        gp = torch.as_tensor(self.helper.transform_all(gp_0, lib=np), dtype=torch.float32)
        return gp, sfh_disk, sfh_bulge


    def _create_tau_matrix(self, age_bins):
        tau_edges = self.lib_ssp.tau_edges.numpy()
        n_edge = len(tau_edges)
        d_tau_base = np.diff(tau_edges)

        matrix = np.zeros([len(age_bins) - 1, len(tau_edges) - 1], dtype=np.float32)
        for i_step in range(len(age_bins) - 1):
            t_lower = age_bins[i_step]
            t_upper = age_bins[i_step + 1]
            dt = t_upper - t_lower
            assert dt > 0, "Age bins must be strictly increasing."
            matrix_sub = np.zeros(len(d_tau_base))
            idx_lower = bisect_left(tau_edges, t_lower)
            idx_upper = bisect_left(tau_edges, t_upper)
            if idx_lower == 0:
                if idx_upper == 0:
                    raise ValueError("An age bin is entirely beyond the youngest age.")
                else:
                    idx_lower = 1
            if idx_upper == n_edge:
                if idx_lower == n_edge:
                    raise ValueError("An age bin is entirely beyond the oldest age.")
                else:
                    idx_upper = n_edge - 1
            if idx_upper == idx_lower:
                # Distribute the mass into one bin
                matrix_sub[idx_lower - 1] = 1.
            elif idx_upper > idx_lower:
                # Distibute the mass into two bins according to the time fraction
                d_tau = np.zeros(len(d_tau_base))
                d_tau[idx_lower : idx_upper-1] = d_tau_base[idx_lower : idx_upper-1]
                d_tau[idx_lower - 1] = (tau_edges[idx_lower] - t_lower)
                d_tau[idx_upper - 1] = (t_upper - tau_edges[idx_upper - 1])
                matrix_sub = d_tau/dt
            else:
                ValueError("Something wrong with the age bins.")
            matrix[i_step] = matrix_sub
        return matrix


    def _derive_sfh(self, sfh_mass, sfh_metal_mass):
        sfh_mass = np.matmul(sfh_mass, self._tau_matrix)
        sfh_metal_mass = np.matmul(sfh_metal_mass, self._tau_matrix)
        sfh_met = self._derive_met(sfh_mass, sfh_metal_mass)
        sfh_mass = sfh_mass[:, None, :]*self._interpolate_met(sfh_met)
        sfh_mass = torch.as_tensor(sfh_mass, dtype=torch.float32)
        sfh_frac, l_norm = self.lib_ssp.mass_to_light(sfh_mass)
        sfh_frac = sfh_frac.flatten(start_dim=1)
        return sfh_frac, l_norm


    def _derive_met(self, sfh_mass, sfh_metal_mass):
        cond = sfh_mass > 0.
        sfh_met = np.zeros_like(sfh_mass)
        sfh_met[cond] = sfh_metal_mass[cond]/sfh_mass[cond]
        return sfh_met


    def _interpolate_met(self, sfh_met):
        met_centres = self.lib_ssp.met.numpy()
        weights = np.zeros([len(sfh_met), len(met_centres), sfh_met.shape[-1]], dtype=np.float32)
        met = np.copy(sfh_met) # Make a copy
        met[met < met_centres[0]] = met_centres[0]
        met[met > met_centres[-1]] = met_centres[-1]
        inds = np.searchsorted(met_centres, met)
        inds[inds == 0] = 1
        met_lower = met_centres[inds - 1]
        met_upper = met_centres[inds]
        weights_lower = (met_upper - met)/(met_upper - met_lower)
        weights_upper = 1 - weights_lower
        for i_tau in range(weights.shape[-1]):
            weights[range(len(met)), inds[:, i_tau] - 1, i_tau] = weights_lower[:, i_tau]
            weights[range(len(met)), inds[:, i_tau], i_tau] = weights_upper[:, i_tau]
        return weights


def simplex_transform(x):
    """Transform a unit hypercube into a simplex.

    If the input is uniformly distributed in (0, 1), the output will follow a
    flat Dirichlet distribution.
    """
    x = -torch.log(1 - x)
    return x/x.sum(dim=-1, keepdim=True)


def compute_interp_weights(x, xp):
    """Compute the weights for lienar interpolation.

    This function assumes that the point at which to evaluate is two
    dimensional.

    Parameters
    ----------
    x : tensor
        (N, M). Points at which to evaluate
    xp : tensor
        (D,). Data points

    Returns
    -------
    weights : tensor
        (N, M, D). Weights for linear interpolation.
    """
    n_interp = xp.size(0)
    n_x = x.size(1)
    x = torch.ravel(x)
    inds = torch.searchsorted(xp, x)
    inds[inds == 0] = 1
    inds[inds == n_interp] = n_interp - 1
    x0 = xp[inds - 1]
    x1 = xp[inds]
    w0 = (x1 - x)/(x1 - x0)
    w1 = 1 - w0
    weights = w0[:, None]*F.one_hot(inds - 1, n_interp) + w1[:, None]*F.one_hot(inds, n_interp)
    weights = weights.reshape(-1, n_x, n_interp)
    return weights

