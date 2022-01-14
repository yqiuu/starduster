from .utils import Configurable, constants

import pickle
from bisect import bisect_left

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Adapter(nn.Module, Configurable):
    """Apply different parametrisations to input parameters.

    Parameters
    ----------
    helper : Helper
        Helper of input parameters.
    lib_ssp : SSPLibrary
        A simple stellar population library.
    selector_disk : Selector
        Selector of the disk component.
    selector_bulge : Selector
        Selector of the bulge component.
    """
    def __init__(self, helper, lib_ssp, selector_disk=None, selector_bulge=None):
        self.helper = helper
        self.lib_ssp = lib_ssp
        nn.Module.__init__(self)
        Configurable.__init__(self,
            pset_gp=GalaxyParameter(),
            pset_sfh_disk=VanillaGrid(),
            pset_sfh_bulge=VanillaGrid(),
            flat_input=False,
            check_sfh_norm=True,
        )
        if selector_disk is not None:
            self.selector_disk = selector_disk
        if selector_bulge is not None:
            self.selector_bulge = selector_bulge
        self.register_buffer("_device", torch.tensor(0.), persistent=False)


    def update_config(self):
        free_shape = []
        param_names = []
        bounds = []
        for name, pset in self.get_config().items():
            if name.startswith('pset'):
                pset.enable(self.helper, self.lib_ssp)
                free_shape.append(pset.input_size)
                param_names.extend(pset.param_names)
                bounds.append(pset.bounds)
        self.free_shape = free_shape
        self.input_size = sum(self.free_shape)
        self.param_names = param_names
        self.bounds = np.vstack(bounds)


    def forward(self, *args, check_bounds=False):
        if self.flat_input:
            params = torch.as_tensor(args[0], dtype=torch.float32, device=self.device)
            params = torch.atleast_2d(params)
            gp, sfh_disk, sfh_bulge = torch.split(params, self.free_shape, dim=-1)
        else:
            gp, sfh_disk, sfh_bulge = args

        if check_bounds:
            # Check if all input parameters are within the bounds
            is_out = torch.full((gp.size(0),), False, device=self.device)
            for val, pset in zip(
                [gp, sfh_disk, sfh_bulge],
                [self.pset_gp, self.pset_sfh_disk, self.pset_sfh_bulge]
            ):
                is_out |= pset.check_bounds(val)
            # Apply the parameterisations
            gp, sfh_disk, sfh_bulge = self._apply_pset(gp, sfh_disk, sfh_bulge)
            # Check if all parameters are in the effective region
            # -Assume helper of selector_disk and selector_bulge are the same
            helper = self.selector_disk.helper
            is_out |= ~self.selector_disk.select(helper.get_item(gp, 'curve_disk_inds'))
            is_out |= ~self.selector_bulge.select(helper.get_item(gp, 'curve_bulge_inds'))
            #
            return gp, sfh_disk, sfh_bulge, is_out
        else:
            return self._apply_pset(gp, sfh_disk, sfh_bulge)


    @property
    def device(self):
        return self._device.device


    def _apply_pset(self, gp, sfh_disk, sfh_bulge):
        gp = self.pset_gp(gp)
        sfh_disk = self.pset_sfh_disk(sfh_disk)
        sfh_bulge = self.pset_sfh_bulge(sfh_bulge)

        if self.check_sfh_norm:
            msg = "Star formation history must be normalised to one."
            assert torch.allclose(sfh_disk.sum(dim=-1), torch.tensor(1.), atol=1e-5), msg
            assert torch.allclose(sfh_bulge.sum(dim=-1), torch.tensor(1.), atol=1e-5), msg

        return gp, sfh_disk, sfh_bulge


class ParameterSet(nn.Module):
    """Base class for a parameter set."""
    def __init__(self, *args):
        super().__init__()
        self._args = args


    def enable(self, helper, lib_ssp):
        param_names, fixed_params, bounds_default, bounds, clip_bounds \
            = self.init(helper, lib_ssp, self._args)
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


    def init(self, helper, lib_ssp, args):
        """
        Returns
        -------
        param_names : list
            List of parameter names.
        fixed_params : dict
            A dictionary to specify fixed parameters. Use the name of the parameter
            as the key.
        bounds_default : array
            An array of (min, max) to specify the default bounds of the parameters.
        bounds : array
            An array of (min, max) to specify the working bounds of the parameters.
        clip_bounds : bool
            If true, when an input value is beyond a bound, set it to be the same
            with the bound.
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
        params = self.set_fixed_params(params)
        params = self.derive_full_params(params)
        return params


    def check_bounds(self, params):
        if self.input_size == 0:
            return torch.full((params.size(0),), False)
        else:
            return torch.any(params <= self.lbounds, dim=-1) \
                | torch.any(params >= self.ubounds, dim=-1)


    def set_fixed_params(self, params):
        params_out = self.params_default.tile((params.size(0), 1))
        params_out[:, self.free_inds] = params
        return params_out


    def derive_full_params(self, params):
        return params


    def _clip_bounds(self, params):
        if self.clip_bounds:
            eps = 1e-6
            params = (params - self.bound_centre)/self.bound_radius
            params = F.hardtanh(params, -1 + eps, 1 - eps)
            params = self.bound_radius*params + self.bound_centre
        return params


class GalaxyParameter(ParameterSet):
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


    def init(self, helper, lib_ssp, args):
        bounds, clip_bounds, fixed_params = args
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


class VanillaGrid(ParameterSet):
    """Discrete star formation history.

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


    def init(self, helper, lib_ssp, args):
        self.simplex_transform , bounds, clip_bounds, fixed_params = args
        param_names = [f'sfr_{i_sfr}' for i_sfr in range(lib_ssp.n_ssp)]
        bounds_default = [(0., 1.)]*lib_ssp.n_ssp
        return param_names, fixed_params, bounds_default, bounds, clip_bounds


    def derive_full_params(self, params):
        if self.simplex_transform:
            return simplex_transform(params)
        else:
            return params


class CompositeGrid(ParameterSet):
    def __init__(self, sfh_model, mh_model, bounds=None, clip_bounds=True, **fixed_params):
        super().__init__(sfh_model, mh_model, bounds, clip_bounds, fixed_params)


    def init(self, helper, lib_ssp, args):
        sfh_model, mh_model, bounds, clip_bounds, fixed_params = args
        param_names_sfh, bounds_default_sfh = sfh_model.enable(lib_ssp)
        param_names_mh, bounds_default_mh = mh_model.enable(lib_ssp)
        param_names = param_names_sfh + param_names_mh
        bounds_default = np.vstack([bounds_default_sfh, bounds_default_mh])
        self.split_size = (len(param_names_sfh), len(param_names_mh))
        self.sfh_model = sfh_model
        self.mh_model = mh_model
        return param_names, fixed_params, bounds_default, bounds, clip_bounds


    def derive_full_params(self, params):
        params_sfh, params_mh = torch.split(params, self.split_size, dim=1)
        sfh = self.sfh_model.derive(params_sfh)
        mh = self.mh_model.derive(params_mh, sfh)
        return torch.flatten(sfh[:, None, :]*mh, start_dim=1)


class SFHComponent(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self._args = args


    def enable(self, lib_ssp):
        param_names, bounds_default = self._init(lib_ssp, *self._args)
        self._args = None
        return param_names, bounds_default


    def _init(self, lib_ssp, *args):
        # return param_names, bounds_default
        pass


    def derive(self, *args):
        pass


class DiscreteSFH(SFHComponent):
    def __init__(self, simplex_transform=False):
        super().__init__(simplex_transform)


    def _init(self, lib_ssp, simplex_transform):
        self.simplex_transform = simplex_transform
        n_tau = lib_ssp.n_tau
        param_names = [f'sfr_{i_sfr}' for i_sfr in range(n_tau)]
        bounds_default = np.tile((0., 1.), (n_tau, 1))
        return param_names, bounds_default


    def derive(self, params):
        if self.simplex_transform:
            params = simplex_transform(params)
        return params


class InterpolatedMH(SFHComponent):
    def _init(self, lib_ssp, *args):
        n_met = len(lib_ssp.met)
        log_met = torch.log10(lib_ssp.met/constants.met_sol)
        self.register_buffer('log_met', log_met)
        #param_names = [f'log_met_{i_met}' for i_met in range(n_met)]
        #bounds_default = np.tile((log_met[0].item(), log_met[-1].item()), (n_met, 1))
        param_names = ['log_met']
        bounds_default = np.array([[log_met[0].item(), log_met[-1].item()]])
        return param_names, bounds_default


    def derive(self, params, sfh):
        # params (N, *) -> (N, N_met, N_age)
        params = params.contiguous()
        return torch.swapaxes(compute_interp_weights(params, self.log_met), 1, 2)


class DiscreteSFR_InterpolatedMet(ParameterSet):
    """Star formation history with discrete stellar age and interpolated
    metallicity.

    Parameters
    ----------
    sed_model : MultiwavelengthSED
        The target SED model.
    sfr_bins : list
        A sequence of index pairs to specifiy the star formation rate bins. If
        not give, use the default bins.
    uni_met : bool
        If true, use the same metallicity for all star formation rate bins.
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
    # TODO: Add functions to make sure that the SFH is normalised to one.
    # Handle the situation where there are only two SFR bins.
    def __init__(self,
        sfr_bins=None, uni_met=False, simplex_transform=False,
        bounds=None, clip_bounds=True, **fixed_params
    ):
        super().__init__(sfr_bins, uni_met, simplex_transform, bounds, clip_bounds, fixed_params)


    def init(self, helper, lib_ssp, args):
        sfr_bins, uni_met, simplex_transform, bounds, clip_bounds, fixed_params = args
        self.n_tau_ssp = lib_ssp.n_tau
        if sfr_bins is None:
            n_sfr = lib_ssp.n_tau
        else:
            n_sfr = len(sfr_bins)
        self.sfr_bins = sfr_bins
        self.n_sfr = n_sfr
        self.simplex_transform = simplex_transform

        log_met = torch.log10(lib_ssp.met/constants.met_sol)
        log_met_min = float(log_met[0])
        log_met_max = float(log_met[-1])
        if uni_met:
            param_names = [f'sfr_{i_sfr}' for i_sfr in range(n_sfr)] + ['log_met']
            bounds_default = np.array([(0., 1.)]*n_sfr + [(log_met_min, log_met_max)])
        else:
            param_names = [f'sfr_{i_sfr}' for i_sfr in range(n_sfr)] \
                + [f'log_met_{i_sfr}' for i_sfr in range(n_sfr)]
            bounds_default = np.array([(0., 1.)]*n_sfr + [(log_met_min, log_met_max)]*n_sfr)
        self.register_buffer('log_met', log_met)
        return param_names, fixed_params, bounds_default, bounds, clip_bounds


    def derive_full_params(self, params):
        sfr = params[:, :self.n_sfr]
        log_met = params[:, self.n_sfr:]
        if self.simplex_transform:
            sfr = simplex_transform(sfr)
        sfh = sfr[:, None, :]*torch.swapaxes(compute_interp_weights(log_met, self.log_met), 1, 2)
        if self.n_sfr != self.n_tau_ssp:
            sfh_full = torch.zeros([sfh.size(0), sfh.size(1), self.n_tau_ssp],
                dtype=sfh.dtype, layout=sfh.layout, device=sfh.device)
            for i_b, (idx_b, idx_e) in enumerate(self.sfr_bins):
                sfh_full[:, :, idx_b:idx_e] = sfh[:, :, i_b, None]/(idx_e - idx_b)
            sfh = sfh_full
        sfh = torch.flatten(sfh, start_dim=1)
        return sfh


    def derive_interp_met(self, log_met):
        n_interp = len(self.log_met)
        n_met = log_met.size(1)
        log_met = torch.ravel(log_met)
        inds = torch.searchsorted(self.log_met, log_met)
        inds[inds == 0] = 1
        inds[inds == n_interp] = n_interp - 1
        met0 = self.log_met[inds - 1]
        met1 = self.log_met[inds]
        w0 = (met1 - log_met)/(met1 - met0)
        w1 = 1 - w0
        weights = w0[:, None]*F.one_hot(inds - 1, n_interp) + w1[:, None]*F.one_hot(inds, n_interp)
        weights = torch.swapaxes(weights.reshape(-1, n_met, n_interp), 1, 2)
        return weights


class InterpolatedSFH(ParameterSet):
    """Star formation history obtained using the invserse distance weighted
    interpolation. Stellar age and metallicity are interpolated independently.

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


    def init(self, helper, lib_ssp, args):
        bounds, clip_bounds, fixed_params = args
        log_met = torch.log10(lib_ssp.met/constants.met_sol)
        log_tau = torch.log10(lib_ssp.tau)
        param_names = ['log_met', 'log_tau']
        bounds_default = np.asarray([
            (log_met[0].item(), log_met[-1].item()), (log_tau[0].item(),log_tau[-1].item()),
        ])
        self.register_buffer('log_met', log_met)
        self.register_buffer('log_tau', log_tau)
        return param_names, fixed_params, bounds_default, bounds, clip_bounds


    def derive_full_params(self, params):
        eps = 1e-6
        log_met, log_tau = params.T
        w_met = torch.swapaxes(compute_interp_weights(log_met[:, None], self.log_met), 1, 2)
        w_tau = compute_interp_weights(log_tau[:, None], self.log_tau)
        return torch.flatten(w_met*w_tau, start_dim=1)


class SemiAnalyticConventer:
    def __init__(self, sed_model, age_bins):
        self.helper = sed_model.helper
        self.lib_ssp = sed_model.lib_ssp
        self._tau_matrix = self._create_tau_matrix(age_bins)


    def __call__(self,
        theta, m_dust, r_dust, r_disk, r_bulge,
        sfh_mass_disk, sfh_metal_mass_disk, sfh_mass_bulge, sfh_metal_mass_bulge
    ):
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

