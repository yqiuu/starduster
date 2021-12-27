import torch
import sedpy
import starduster


def test_integration():
    torch.set_num_threads(1)
    torch.manual_seed(831)
    # Create SED model
    band_names = [
        'galex_FUV', 'galex_NUV',
        'sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0',
        'twomass_J', 'twomass_H', 'twomass_Ks',
        'wise_w1', 'wise_w2', 'wise_w3', 'wise_w4',
        'herschel_pacs_100', 'herschel_pacs_160',
        'herschel_spire_250', 'herschel_spire_350', 'herschel_spire_500'
    ]
    filters = sedpy.observate.load_filters(band_names)
    z_test = 0.13
    distmod= 0.

    sed_model = starduster.MultiwavelengthSED.from_builtin()
    sed_model.configure_detector(filters, redshift=z_test, distmod=distmod, ab_mag=True)
    sed_model.configure_adapter(
        starduster.GalaxyParameter(sed_model, bounds={'b_to_t':(.1, .8)}),
        starduster.InterpolatedSFH(sed_model),
        starduster.InterpolatedSFH(sed_model),
        flat_input=True,
    )
    # Create test data
    eps = 0.05
    eps_post = .1
    x_test = starduster.sample_effective_region(sed_model)
    with torch.no_grad():
        y_test = sed_model(x_test, return_ph=True)
        y_err = torch.full_like(y_test, eps)
    # Create posterior
    likelihood = starduster.Gaussian(y_test, y_err, norm=False)
    posterior = starduster.Posterior(sed_model, likelihood)
    posterior.configure_output_mode(output_mode='torch', negative=True, log_out=0.)
    # Fit the test data
    sampler = lambda n_samp: x_test*(1 + eps*(2*torch.rand(n_samp, len(x_test)) - 1))
    x0 = starduster.sample_effective_region(sed_model, sampler=sampler)
    x_pred = starduster.optimize(posterior, torch.optim.Adam, x0, n_step=300, lr=1e-2)
    # Check data
    with torch.no_grad():
        log_post = float(posterior(x_pred))
    assert log_post < eps_post

