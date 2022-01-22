import pytest
import torch
import starduster as sd
import numpy as np
from numpy import testing


torch.set_num_threads(1)
torch.manual_seed(831)
RTOL = 1.3e-6
ATOL = 1e-5


def sample_params(n_samp, bounds):
    lb, ub = torch.as_tensor(bounds, dtype=torch.float32).T
    return ub + (lb - ub)*torch.rand(n_samp, len(lb))


@pytest.mark.parametrize("target,pn_type,need_check_norm", (
    (sd.GalaxyParameter(), 'gp', False),
    (sd.VanillaGrid(simplex_transform=True), 'sfh', True),
    (sd.CompositeGrid(sd.ExponentialSFH(), sd.InterpolatedMH()), 'sfh', True)
))
def test_parametrization(target, pn_type, need_check_norm):
    # Test whether the parameterization gives the correct shape and whether the
    # derived SFH is normalised to one.
    sed_model = sd.MultiwavelengthSED.from_builtin()
    target.enable(sed_model.helper, sed_model.lib_ssp)

    n_samp = 5
    params_in = sample_params(n_samp, target.bounds)
    params_out = target(params_in)
    
    if pn_type == 'gp':
        n_param = len(sed_model.helper.header)
    elif pn_type == 'sfh':
        n_param = sed_model.lib_ssp.n_ssp
    else:
        raise ValueError(f"Unkown 'pn_type': {pn_type}.")
    assert params_out.size() == torch.Size((n_samp, n_param))
    
    if need_check_norm:
        testing.assert_allclose(torch.sum(params_out, dim=1).numpy(), 1., rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("target,pn_type", (
    (sd.GalaxyParameter, 'gp'),
    (sd.VanillaGrid, 'sfh'),
))
def test_fixed_params(target, pn_type):
    # Test whether the parameterization works when some parameters are fixed.
    sed_model = sd.MultiwavelengthSED.from_builtin()
    helper = sed_model.helper
    pn = target()
    pn.enable(helper, sed_model.lib_ssp)
    # Test the case where one parameter is fixed
    idx = torch.randint(0, len(pn.param_names), (1,)).item()
    lb, ub = pn.bounds[idx]
    fixed_value = .5*(lb + ub)
    if pn_type == 'gp':
        key = pn.param_names[idx]
        fixed_params = {key: helper.recover(fixed_value, key)}
    elif pn_type == 'sfh':
        fixed_params = {pn.param_names[idx]: fixed_value}
    else:
        raise ValueError(f"Unkown 'pn_type': {pn_type}.")
    pn_fixed = target(**fixed_params)
    pn_fixed.enable(sed_model.helper, sed_model.lib_ssp)

    n_samp = 5
    params_in = sample_params(n_samp, pn_fixed.bounds)
    params_in_expect = np.insert(params_in.numpy(), idx, fixed_value, axis=1)
    params_in_expect = torch.as_tensor(params_in_expect)
    params_out = pn_fixed(params_in).numpy()
    params_out_expect = pn(params_in_expect).numpy()
    testing.assert_allclose(params_out, params_out_expect, rtol=RTOL, atol=ATOL)
    # Test the case where all parameters are fixed
    params_all_fixed = torch.as_tensor(pn.bounds.mean(axis=1), dtype=torch.float32)
    if pn_type == 'gp':
        fixed_params = {
            key: helper.recover(val, key) for key, val in zip(pn.param_names, params_all_fixed)
        }
    else:
        fixed_params = {
            key: val for key, val in zip(pn.param_names, params_all_fixed)
        }
    pn_all_fixed = target(**fixed_params)
    pn_all_fixed.enable(sed_model.helper, sed_model.lib_ssp)

    params_in = torch.zeros([n_samp, 0])
    params_out = pn_all_fixed(params_in).numpy()
    params_out_expect = params_all_fixed.tile((n_samp, 1)).numpy()
    testing.assert_allclose(params_out, params_out_expect, rtol=RTOL, atol=ATOL)

