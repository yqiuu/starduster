import os
import pickle

import numpy as np
from numpy import testing
import torch
import sedpy
import starduster


def test_sed_model():
    # Test whether the SED model can reproduce the stored data.
    sed_model = create_sed_model()
    data = pickle.load(open(get_fname(), "rb"))
    params = data['inputs']

    for kwargs, res_fid in zip(data['options'], data['spectra']):
        with torch.no_grad():
            testing.assert_allclose(
                res_fid, sed_model(*params, **kwargs).numpy(), rtol=1e-4, atol=1e-4
            )

    with torch.no_grad():
        dust_props_test = predict_dust_props(sed_model, params)
    for arr_fid, arr_test in zip(data['dust_props'], dust_props_test):
        testing.assert_allclose(arr_fid, arr_test, rtol=1e-4, atol=1e-4)

    # Test flat input
    sed_model.configure(flat_input=True)
    params_flat = torch.hstack(params)
    for kwargs, arr_fid in zip(data['options'], data['spectra']):
        with torch.no_grad():
            testing.assert_allclose(
                arr_fid, sed_model(params_flat, **kwargs).numpy(), rtol=1e-4, atol=1e-4
            )
        # Only test one item
        break

    # Test raises
    testing.assert_raises(ValueError, sed_model, params_flat, component='???')
    testing.assert_raises(ValueError, sed_model.configure, xxx=None)


def create_test_data():
    sed_model = create_sed_model()
    input_shape = sed_model.adapter.free_shape
    rstate = np.random.RandomState(831)
    n_samp = 2
    gp = torch.tensor(rstate.uniform(-1, 1, [n_samp, input_shape[0]]), dtype=torch.float32)
    sfh_disk = torch.tensor(
    rstate.dirichlet(np.full(input_shape[1], 0.5), n_samp), dtype=torch.float32
    )
    sfh_bulge = torch.tensor(
        rstate.dirichlet(np.full(input_shape[2], 0.5), n_samp), dtype=torch.float32
    )

    options = [
        {'return_ph': False, 'return_lum': False},
        {'return_ph': False, 'return_lum': True},
        {'return_ph': False, 'return_lum': False, 'component': 'dust_emission'},
        {'return_ph': True,},
    ]
    spectra = []
    with torch.no_grad():
        for kwargs in options:
            spectra.append(sed_model(gp, sfh_disk, sfh_bulge, **kwargs).numpy())

        dust_props = predict_dust_props(sed_model, (gp, sfh_disk, sfh_bulge))

    data = {
        'options': options,
        'inputs': (gp, sfh_disk, sfh_bulge),
        'spectra': spectra,
        'dust_props': dust_props,
    }
    pickle.dump(data, open(get_fname(), "wb"))


def predict_dust_props(sed_model, params):
    return (
        sed_model.predict_absorption_fraction(*params, return_lum=True).numpy(),
        sed_model.predict_attenuation(*params, windows=[(5450., 5550.)]).numpy(),
        sed_model.predict_attenuation(*params).numpy(),
    )


def create_sed_model():
    band_names = [
        'galex_FUV', 'galex_NUV',
        'sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0',
        'twomass_J', 'twomass_H', 'twomass_Ks',
        'wise_w1', 'wise_w2', 'wise_w3', 'wise_w4',
        'herschel_pacs_100', 'herschel_pacs_160',
        'herschel_spire_250', 'herschel_spire_350', 'herschel_spire_500'
    ]
    filters = sedpy.observate.load_filters(band_names)
    sed_model = starduster.MultiwavelengthSED.from_builtin('base')
    sed_model.configure(filters=filters)
    return sed_model



def get_fname():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_sed_model.pickle")


if __name__ == "__main__":
    create_test_data()

