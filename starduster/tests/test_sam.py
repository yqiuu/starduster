import os
import pickle

import numpy as np
import torch
import starduster


RTOL = 1.3e-6
ATOL = 1e-5


def test_converter():
    # Test whether the converter can reproduce the stored data.
    data = pickle.load(open(get_fname(), "rb"))
    converter, _ = create_converter()
    gp_fid = data.pop('gp')
    sfh_disk_fid = data.pop('sfh_disk')
    sfh_bulge_fid = data.pop('sfh_bulge')
    gp_test, sfh_disk_test, sfh_bulge_test = converter(**data)
    np.testing.assert_allclose(gp_fid.numpy(), gp_test.numpy(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(sfh_disk_fid.numpy(), sfh_disk_test.numpy(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(sfh_bulge_fid.numpy(), sfh_bulge_test.numpy(), rtol=RTOL, atol=ATOL)


def create_test_data():
    torch.set_num_threads(1)
    rstate = np.random.RandomState(831)

    converter, age_bins = create_converter()

    n_gal = 10
    # Sample some galaxy properties
    theta = rstate.uniform(0, 90, n_gal)
    m_dust = 10**rstate.uniform(6, 8, n_gal)
    r_dust = 10**rstate.uniform(-2, 2.5, n_gal)
    r_disk = 10**rstate.uniform(-2, 2.5, n_gal)
    r_bulge = 10**rstate.uniform(-0.5, 1.5, n_gal)
    # Sample some star formation histories
    n_sfh = len(age_bins) - 1
    sfh_mass_disk = 10**rstate.uniform(2, 8, [n_gal, n_sfh])
    sfh_mass_bulge = 10**rstate.uniform(2, 8, [n_gal, n_sfh])
    sfh_metal_mass_disk = 10**rstate.uniform(-2, 4, [n_gal, n_sfh])
    sfh_metal_mass_bulge = 10**rstate.uniform(-2, 4, [n_gal, n_sfh])
    # Compute spectra
    gp, sfh_disk, sfh_bulge = converter(
        theta=theta,
        m_dust=m_dust,
        r_dust=r_dust,
        r_disk=r_disk,
        r_bulge=r_bulge,
        sfh_mass_disk=sfh_mass_disk,
        sfh_metal_mass_disk=sfh_metal_mass_disk,
        sfh_mass_bulge=sfh_mass_bulge,
        sfh_metal_mass_bulge=sfh_metal_mass_bulge
    )
    data = {
        'theta': theta, 'm_dust': m_dust,
        'r_dust': r_dust, 'r_disk': r_disk, 'r_bulge': r_bulge,
        'sfh_mass_disk': sfh_mass_disk, 'sfh_metal_mass_disk': sfh_metal_mass_disk,
        'sfh_mass_bulge': sfh_mass_bulge, 'sfh_metal_mass_bulge': sfh_metal_mass_bulge,
        'gp': gp, 'sfh_disk': sfh_disk, 'sfh_bulge': sfh_bulge
    }
    pickle.dump(data, open(get_fname(), "wb"))


def create_converter():
    n_sfh = 10
    age_bins = np.logspace(5.8, 10.5, n_sfh + 1)
    sed_model = starduster.MultiwavelengthSED.from_builtin()
    converter = starduster.SemiAnalyticConventer(sed_model, age_bins)
    return converter, age_bins


def get_fname():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_sam.pickle")


if __name__ == "__main__":
     create_test_data()

