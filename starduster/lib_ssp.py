from .utils import reduction, interp_arr

import pickle
from os import path

import numpy as np
import torch


class SSPLibrary:
    """Simple stellar population library.

    The SSP library should be a dictionary stored as a pickle file. The
    dictionary should have the following keys:
    - lam: Wavelength. [micron]
    - met: Metallicity. [dimensionless]
    - tau: Stellar age. [yr]
    - flx: (lam, met, age). Time averaged SSP spectrum. [L_sol/micron]
    - norm: (met, age). Normalization of the spectrum. [L_sol]
    - tau_edges: Stellar age bins of the integration. [yr]

    Parameters
    ----------
    fname : str
        File name of the SSP library.
    lam_base : array
        Wavelength grid that is used in the radiative simulation.
    regrid : str
        Option to choose a different wavelength grid. See
        MultiwavelengthSED.from_builtin.
    eps_reduce : float
        Tolerance parameter for the reduced spectrum.
    """
    def __init__(self, fname, lam_base, regrid, eps_reduce):
        lib_ssp = pickle.load(open(fname, "rb"))
        lam_ssp = lib_ssp['lam'] # mircon
        log_lam_ssp = np.log(lam_ssp)
        l_ssp_raw = lib_ssp['flx']/lib_ssp['norm']
        self.sfh_shape = l_ssp_raw.shape[1:] # (lam, met, age)
        self.n_met = len(lib_ssp['met'])
        self.n_tau = len(lib_ssp['tau'])
        self.n_ssp = self.n_met*self.n_tau
        self.dim_age = 2
        self.dim_met = 1
        l_ssp_raw.resize(l_ssp_raw.shape[0], l_ssp_raw.shape[1]*l_ssp_raw.shape[2])
        l_ssp_raw = l_ssp_raw.T
        l_ssp_raw *= lam_ssp
        L_ssp = reduction(l_ssp_raw, log_lam_ssp, eps=eps_reduce)[0]
        lam_eval = self.prepare_lam_eval(regrid, l_ssp_raw, log_lam_ssp, lam_base, lam_ssp)
        l_ssp = interp_arr(np.log(lam_eval), log_lam_ssp, l_ssp_raw, right=0.)
        # Save attributes
        self.tau = torch.tensor(lib_ssp['tau'], dtype=torch.float32)
        self.met = torch.tensor(lib_ssp['met'], dtype=torch.float32)
        self.lam_base = torch.tensor(lam_base, dtype=torch.float32)
        self.lam_eval = torch.tensor(lam_eval, dtype=torch.float32)
        self.l_ssp = torch.tensor(l_ssp, dtype=torch.float32)
        self.L_ssp = torch.tensor(L_ssp, dtype=torch.float32)
        self.norm = torch.tensor(lib_ssp['norm'], dtype=torch.float32)
        self.tau_edges = torch.tensor(lib_ssp['tau_edges'], dtype=torch.float32)
        self.d_tau = torch.diff(self.tau_edges)


    @classmethod
    def from_builtin(cls, regrid='auto', eps_reduce=4e-5):
        dirname = path.join(path.dirname(path.abspath(__file__)), "data")
        fname = path.join(dirname, "FSPS_Chabrier_neb_compact.pickle")
        lam_base = pickle.load(open(path.join(dirname, "lam_main.pickle"), "rb"))
        return cls(fname, lam_base, regrid, eps_reduce)


    def prepare_lam_eval(self, regrid, l_ssp_raw, log_lam_ssp, lam_base, lam_full):
        self.regrid = regrid
        if regrid == 'base':
            lam_eval = lam_base
        elif regrid == 'auto':
            inds_reduce = reduction(l_ssp_raw, log_lam_ssp, eps=1e-5)[-1]
            lam_reduce = lam_full[inds_reduce]
            lam_eval = np.append(lam_reduce[lam_reduce >= lam_base[0]], lam_base)
            lam_eval = np.sort(lam_eval)
        elif regrid == 'full':
            lam_eval = np.append(lam_full[lam_full >= lam_base[0]], lam_base)
            lam_eval = np.sort(lam_eval)
        else:
            lam_eval = np.asarray(lam_eval)
        return lam_eval


    def reshape_sfh(self, sfh):
        """Convert flattened star formation history into 2D grid.

        Parameters
        ----------
        sfh : tensor
            (N, D_met*D_age).

        Returns
        -------
        tensor
            (N, D_met, D_age).
        """
        return torch.atleast_2d(sfh).reshape((-1, *self.sfh_shape))


    def sum_over_age(self, sfh):
        return self.reshape_sfh(sfh).sum(dim=self.dim_age)


    def sum_over_met(self, sfh):
        return self.reshape_sfh(sfh).sum(dim=self.dim_met)


    def mass_to_light(self, sfh_mass):
        """Transform mass to light.

        Parameters
        ----------
        sfh_mass : tensor, (N, D_met, D_age), [M_sol]
            Star formation history.

        Returns
        -------
        sfh_frac : tensor, (N, D_met, D_age)
            Fractional luminosity.
        l_norm : tensor, (N,), [L_sol]
            Intrinsic bolometic luminosity.
        """
        sfh_light = sfh_mass*self.norm
        l_norm = sfh_light.sum(dim=(1, 2))
        sfh_frac = sfh_light/l_norm[:, None, None]
        sfh_frac[l_norm == 0.] = 1./(sfh_frac.size(1)*sfh_frac.size(2))
        return sfh_frac, l_norm

