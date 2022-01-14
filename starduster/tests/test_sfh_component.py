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


def test_discrete_sfh_raises():
    # Test whether DiscreteSFH can find invalid sfh_bins
    torch.set_num_threads(1)
    lib_ssp = starduster.SSPLibrary.from_builtin()

    def init_sfh(sfh_bins):
        starduster.DiscreteSFH(sfh_bins).enable(lib_ssp)
    
    # Indices must be within [0, n_tau]
    testing.assert_raises(AssertionError, init_sfh, [(1, lib_ssp.n_tau + 5)])
    testing.assert_raises(AssertionError, init_sfh, [(-2, 1)])
    # Avoid flips 
    testing.assert_raises(AssertionError, init_sfh, [(2, 1)])
    # Avoid overlaps
    testing.assert_raises(AssertionError, init_sfh, [(0, 2), (1, 3)])

