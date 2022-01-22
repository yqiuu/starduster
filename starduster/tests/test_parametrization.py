import pytest
import torch
import starduster as sd
from numpy import testing


torch.set_num_threads(1)
torch.manual_seed(831)


@pytest.mark.parametrize("target,pn_type,need_check_norm", (
    (sd.GalaxyParameter(), 'gp', False),
    (sd.VanillaGrid(simplex_transform=True), 'sfh', True),
    (sd.CompositeGrid(sd.ExponentialSFH(), sd.InterpolatedMH()), 'sfh', True)
))
def test_parametrization(target, pn_type, need_check_norm):
    sed_model = sd.MultiwavelengthSED.from_builtin()
    target.enable(sed_model.helper, sed_model.lib_ssp)

    n_samp = 5
    lb, ub = torch.as_tensor(target.bounds, dtype=torch.float32).T
    params_in = ub + (lb - ub)*torch.rand(n_samp, len(lb))
    params_out = target(params_in)
    
    if pn_type == 'gp':
        n_param = len(sed_model.helper.header)
    elif pn_type == 'sfh':
        n_param = sed_model.lib_ssp.n_ssp
    else:
        raise ValueError(f"Unkown 'pn_type': {pn_type}.")
    assert params_out.size() == torch.Size((n_samp, n_param))
    
    if need_check_norm:
        testing.assert_allclose(torch.sum(params_out, dim=1).numpy(), 1., rtol=1.3e-6, atol=1e-5)

