from .utils import constants

import pickle
from bisect import bisect_left

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Adapter(nn.Module):
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
        super().__init__()
        self.helper = helper
        self.lib_ssp = lib_ssp
        if selector_disk is not None:
            self.selector_disk = selector_disk
        if selector_bulge is not None:
            self.selector_bulge = selector_bulge
        self.register_buffer("_device", torch.tensor(0.), persistent=False)
        self.configure(
            GalaxyParameter(self), DiscreteSFH(self), DiscreteSFH(self),
            flat_input=False, check_sfh_norm=True
        )


    def forward(self, *args, check_bounds=False):
        if self.flat_input:
            params = torch.as_tensor(args[0], dtype=torch.float32, device=self.device)
            gp, sfh_disk, sfh_bulge = self.unflatten(params)
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


    def configure(
        self, gp=None, sfh_disk=None, sfh_bulge=None, flat_input=None, check_sfh_norm=None
    ):
        """Configure the input mode.

        Parameters
        ----------
        gp : ParameterSet
            Parametrisation of the galaxy parameters.
        sfh_disk : ParameterSet
            Parametrisation of the disk star formation history.
        sfh_bulge : ParameterSet
            Parametrisation of the bulge star formation history.
        flat_input : bool
            If ``True``, assume the input array is flat.
        check_sfh_norm : bool
            If ``True``, raise an error when star formation history is not
            normalised to one.
        """
        if gp is not None:
            self.pset_gp = gp
        if sfh_disk is not None:
            self.pset_sfh_disk = sfh_disk
        if sfh_bulge is not None:
            self.pset_sfh_bulge = sfh_bulge
        if flat_input is not None:
            self.flat_input = flat_input
        if check_sfh_norm is not None:
            self.check_sfh_norm = check_sfh_norm
        #
        pset_names = ['pset_gp', 'pset_sfh_disk', 'pset_sfh_bulge']
        free_shape = []
        param_names = []
        bounds = []
        for name in pset_names:
            pset = getattr(self, name)
            free_shape.append(pset.input_size)
            param_names.extend(pset.param_names)
            bounds.append(pset.bounds)
        self.free_shape = free_shape
        self.input_size = sum(self.free_shape)
        self.param_names = param_names
        self.bounds = np.vstack(bounds)


    def unflatten(self, params):
        params = torch.atleast_2d(params)
        params_out = [None]*len(self.free_shape)
        idx_b = 0
        for i_input, size in enumerate(self.free_shape):
            idx_e = idx_b + size
            params_out[i_input] = params[:, idx_b:idx_e]
            idx_b = idx_e
        return params_out


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


    def _get_config(self):
        return {
            'gp': self.pset_gp,
            'sfh_disk' : self.pset_sfh_disk,
            'sfh_bulge': self.pset_sfh_bulge,
            'flat_input': self.flat_input,
            'check_sfh_norm': self.check_sfh_norm,
        }


class ParameterSet(nn.Module):
    """Base class for a parameter set.

    Parameters
    ----------
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
    def __init__(self, param_names, fixed_params, bounds_default, bounds, clip_bounds):
        super().__init__()
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


    def forward(self, params):
        if self.clip_bounds:
            eps = 1e-6
            params = (params - self.bound_centre)/self.bound_radius
            params = F.hardtanh(params, -1 + eps, 1 - eps)
            params = self.bound_radius*params + self.bound_centre

        params = self.derive_full_params(self.set_fixed_params(params))
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
    def __init__(self, sed_model, bounds=None, clip_bounds=True, **fixed_params):
        helper = sed_model.helper
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
        super().__init__(param_names, fixed_params, bounds_default, bounds, clip_bounds)


class DiscreteSFH(ParameterSet):
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
    def __init__(self,
        sed_model, simplex_transform=False, bounds=None, clip_bounds=True, **fixed_params
    ):
        lib_ssp = sed_model.lib_ssp
        self.simplex_transform = simplex_transform
        param_names = [f'sfr_{i_sfr}' for i_sfr in range(lib_ssp.n_ssp)]
        bounds_default = [(0., 1.)]*lib_ssp.n_ssp
        super().__init__(param_names, fixed_params, bounds_default, bounds, clip_bounds)


    def derive_full_params(self, params):
        if self.simplex_transform:
            return simplex_transform(params)
        else:
            return params


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
        sed_model, sfr_bins=None, uni_met=False, simplex_transform=False,
        bounds=None, clip_bounds=True, **fixed_params
    ):
        lib_ssp = sed_model.lib_ssp
        self.n_tau_ssp = lib_ssp.n_tau
        if sfr_bins is None:
            n_sfr = lib_ssp.n_tau
        else:
            n_sfr = len(sfr_bins)
        self.sfr_bins = sfr_bins
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
        super().__init__(param_names, fixed_params, bounds_default, bounds, clip_bounds)
        self.register_buffer('log_met', log_met)
        self.n_sfr = n_sfr


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
    def __init__(self, sed_model, bounds=None, clip_bounds=True, **fixed_params):
        lib_ssp = sed_model.lib_ssp
        log_met = torch.log10(lib_ssp.met/constants.met_sol)
        log_tau = torch.log10(lib_ssp.tau)
        param_names = ['log_met', 's_met', 'log_tau', 's_tau']
        bounds_default = np.asarray([
            (float(log_met[0]), float(log_met[-1])), (-2., 1.),
            (float(log_tau[0]), float(log_tau[-1])), (-2., 1.)
        ])
        super().__init__(param_names, fixed_params, bounds_default, bounds, clip_bounds)
        self.register_buffer('log_met', log_met)
        self.register_buffer('log_tau', log_tau)


    def derive_full_params(self, params):
        eps = 1e-6
        log_met, s_met, log_tau, s_tau = params.T
        w_met = torch.swapaxes(compute_interp_weights(log_met[:, None], self.log_met), 1, 2)
        w_tau = compute_interp_weights(log_tau[:, None], self.log_tau)
        return torch.flatten(w_met*w_tau, start_dim=1)


class SemiAnalyticConventer:
    def __init__(self, sed_model, timesteps):
        self.helper = sed_model.helper
        self.lib_ssp = sed_model.lib_ssp
        self._tau_matrix = self._create_tau_matrix(timesteps)
        

    def __call__(self,
        theta, den_dust, r_dust_to_rd, r_disk, r_bulge,
        sfh_disk_mass, sfh_disk_metal_mass, sfh_bulge_mass, sfh_bulge_metal_mass
    ):
        sfh_disk, l_norm_disk = self._derive_sfh(sfh_disk_mass, sfh_disk_metal_mass)
        sfh_bulge, l_norm_bulge = self._derive_sfh(sfh_bulge_mass, sfh_bulge_metal_mass)
        l_norm = l_norm_disk + l_norm_bulge
        b_to_t = l_norm_bulge/l_norm
        gp_0 = np.vstack([theta, den_dust, r_dust_to_rd, r_disk, r_bulge, l_norm, b_to_t]).T
        gp = torch.as_tensor(self.helper.transform_all(gp_0, lib=np), dtype=torch.float32)
        return gp, sfh_disk, sfh_bulge


    def _create_tau_matrix(self, timesteps):
        tau_edges = self.lib_ssp.tau_edges.numpy()
        d_tau_base = np.diff(tau_edges)

        matrix = np.zeros([len(timesteps) - 1, len(tau_edges) - 1], dtype=np.float32)
        for i_step in range(len(timesteps) - 1):
            t_lower = timesteps[i_step]
            t_upper = timesteps[i_step + 1]
            dt = t_upper - t_lower
            matrix_sub = np.zeros(len(d_tau_base))
            idx_lower = bisect_left(tau_edges, t_lower)
            idx_upper = bisect_left(tau_edges, t_upper)
            if idx_lower == 0:
                if idx_upper == 0:
                    raise ValueError("One time step is below the lower limit.")
                else:
                    idx_lower = 1
            if idx_lower == len(tau_edges) and idx_upper == len(tau_edges):
                raise ValueError("One time step is above the upper limit.")
            if idx_upper == idx_lower:
                matrix_sub[idx_lower - 1] = 1.
            elif idx_upper > idx_lower:
                d_tau = np.zeros(len(d_tau_base))
                d_tau[idx_lower : idx_upper-1] = d_tau_base[idx_lower : idx_upper-1]
                d_tau[idx_lower - 1] = (tau_edges[idx_lower] - t_lower)
                d_tau[idx_upper - 1] = (t_upper - tau_edges[idx_upper - 1])
                matrix_sub = d_tau/dt
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
