import pytest
import torch
import starduster
from numpy import testing


@pytest.mark.parametrize("target", (
    starduster.DiscreteSFH(simplex_transform=True),
    starduster.DiscreteSFH(sfh_bins=[(2, 5)], simplex_transform=True),
    starduster.DiscreteSFH(sfh_bins=[(1, 2), (3, 6)], simplex_transform=True),
    starduster.DiscreteSFH(sfh_bins=[(0, 1), (1, 2), (3, 6)], simplex_transform=True),
))
def test_sfh(target):
    # Test whether the SFH component gives the correct shape and whether the
    # derived SFH is normalised to one.
    torch.set_num_threads(1)
    torch.manual_seed(831)
    lib_ssp = starduster.SSPLibrary.from_builtin()

    n_samp = 5
    param_names, bounds_default = target.enable(lib_ssp)
    lb, ub = torch.as_tensor(bounds_default).T
    params = ub + (lb - ub)*torch.rand(n_samp, len(param_names))
    sfh = target.derive(params)

    assert sfh.size() == torch.Size((n_samp, lib_ssp.n_tau))
    testing.assert_allclose(torch.sum(sfh, dim=1).numpy(), 1.)


@pytest.mark.parametrize("target", (
    starduster.InterpolatedMH(),
))
def test_mh(target):
    # Test whether the SFH component gives the correct shape and whether the
    # derived SFH is normalised to one.
    torch.set_num_threads(1)
    torch.manual_seed(831)
    lib_ssp = starduster.SSPLibrary.from_builtin()

    n_samp = 5
    param_names, bounds_default = target.enable(lib_ssp)
    lb, ub = torch.as_tensor(bounds_default).T
    params = ub + (lb - ub)*torch.rand(n_samp, len(param_names))
    sfh = torch.rand(n_samp, lib_ssp.n_tau)
    sfh = sfh/sfh.sum(dim=1, keepdim=True)
    mh = target.derive(params, sfh)

    assert mh.size() == torch.Size((n_samp, lib_ssp.n_met, 1)) \
        or mh.size() == torch.Size((n_samp, lib_ssp.n_met, lib_ssp.n_tau))
    testing.assert_allclose(torch.sum(mh, dim=(1, 2)).numpy(), 1.)

