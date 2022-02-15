import os
import pickle

import numpy as np
import torch
import starduster


def test_analyzer():
    # Test whether the analyzer can reproduce the stored data.
    data = pickle.load(open(get_fname(), "rb"))
    params = data['input']
    summary_fid = data['output']
    sed_model, analyzer = create_analyzer()

    def test(props, output_type='numpy'):
        summary_test = analyzer.compute_property_summary(params, props, output_type=output_type)
        for name in props:
            arr_fid = summary_fid[name]
            arr_test = summary_test[name]
            if output_type == 'torch':
                assert isinstance(arr_test, torch.Tensor)
                arr_test = arr_test.numpy()
            np.testing.assert_allclose(arr_fid, arr_test, rtol=1e-4, atol=1e-4)

    props_all = analyzer.list_available_properties()
    test(props_all)
    test(props_all, 'torch')
    # Only include disk properties
    props_disk = [name for name in props_all if name.endswith('disk')]
    test(props_disk)
    # Only include bulge properties
    props_bulge = [name for name in props_all if name.endswith('bulge')]
    test(props_bulge)


def create_test_data():
    torch.set_num_threads(1)
    torch.manual_seed(831)
    sed_model, analyzer = create_analyzer()
    n_samp = 10
    params = starduster.sample_effective_region(sed_model, n_samp)
    summary = analyzer.compute_property_summary(
        params, analyzer.list_available_properties(), output_type="numpy"
    )
    data = {'input': params, 'output': summary}
    pickle.dump(data, open(get_fname(), "wb"))


def create_analyzer():
    sed_model = starduster.MultiwavelengthSED.from_builtin('base')
    sed_model.configure(flat_input=True, check_sfh_norm=False)
    analyzer = starduster.Analyzer(sed_model)
    return sed_model, analyzer


def get_fname():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_analyzer.pickle")


if __name__ == "__main__":
    create_test_data()

