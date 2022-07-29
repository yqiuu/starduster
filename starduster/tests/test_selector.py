from numpy import testing
import torch
import starduster


def test_sample_from_selector():
    # Test whether the samples are within the effective region.
    sed_model = starduster.MultiwavelengthSED.from_builtin()
    selector_disk = sed_model.adapter.selector_disk
    selector_bulge = sed_model.adapter.selector_bulge
    n_samp = 5

    # Test selector_disk only
    samps = starduster.sample_from_selector(n_samp, selector_disk, None)
    assert torch.all(selector_disk.select(sed_model.helper.get_item(samps, 'curve_disk_inds')))

    # Test selector_bulge only
    samps = starduster.sample_from_selector(n_samp, None, selector_bulge)
    assert torch.all(selector_bulge.select(sed_model.helper.get_item(samps, 'curve_bulge_inds')))

    # Test both
    samps = starduster.sample_from_selector(n_samp, selector_disk, selector_bulge)
    assert torch.all(selector_disk.select(sed_model.helper.get_item(samps, 'curve_disk_inds'))
        & selector_bulge.select(sed_model.helper.get_item(samps, 'curve_bulge_inds')))
    
    # Test raises
    testing.assert_raises(ValueError, starduster.sample_from_selector, n_samp)

