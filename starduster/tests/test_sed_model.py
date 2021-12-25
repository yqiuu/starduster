import os
import pickle

import numpy as np
import torch
import starduster


def test_sed_model():
    # Test whether the SED model can reproduce the stored data.
    data = pickle.load(open(get_fname(), "rb"))
    params = data['input']
    spectra_fid = data['output']
    sed_model = create_sed_model()
    with torch.no_grad():
        spectra_test = sed_model(*params)
    np.testing.assert_allclose(spectra_fid.numpy(), spectra_test.numpy(), rtol=1e-4, atol=1e-4)


def create_test_data():
    sed_model = create_sed_model()
    input_shape = sed_model.adapter.free_shape
    rstate = np.random.RandomState(831)
    n_samp = 10
    gp = torch.tensor(rstate.uniform(-1, 1, [n_samp, input_shape[0]]), dtype=torch.float32)
    sfh_disk = torch.tensor(
    rstate.dirichlet(np.full(input_shape[1], 0.5), n_samp), dtype=torch.float32
    )
    sfh_bulge = torch.tensor(
        rstate.dirichlet(np.full(input_shape[2], 0.5), n_samp), dtype=torch.float32
    )

    with torch.no_grad():
        spectra = sed_model(gp, sfh_disk, sfh_bulge)

    data = {'input': (gp, sfh_disk, sfh_bulge), 'output': spectra}
    pickle.dump(data, open(get_fname(), "wb"))


def create_sed_model():
    return starduster.MultiwavelengthSED.from_builtin()


def get_fname():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_sed_model.pickle")


if __name__ == "__main__":
    create_test_data()

