from .utils import reduction, interp_arr

import pickle
from os import path

import numpy as np
import torch


class SSPLibrary:
    def __init__(self, fname, lam, eps=5e-4):
        lib_ssp = pickle.load(open(fname, "rb"))
        lam_ssp = lib_ssp['lam']
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
        L_ssp = reduction(l_ssp_raw, log_lam_ssp, eps=eps)[0]
        l_ssp = interp_arr(np.log(lam), log_lam_ssp, l_ssp_raw, right=0.)
        # Save attributes
        self.tau = torch.tensor(lib_ssp['tau'], dtype=torch.float32)
        self.met = torch.tensor(lib_ssp['met'], dtype=torch.float32)
        self.lam = torch.tensor(lam, dtype=torch.float32)
        self.l_ssp = torch.tensor(l_ssp, dtype=torch.float32)
        self.L_ssp = torch.tensor(L_ssp, dtype=torch.float32)
        self.norm = torch.tensor(lib_ssp['norm'], dtype=torch.float32)
        self.tau_edges = torch.tensor(lib_ssp['tau_edges'], dtype=torch.float32)
        self.d_tau = torch.diff(self.tau_edges)


    @classmethod
    def from_builtin(cls):
        dirname = path.join(path.dirname(path.abspath(__file__)), "models")
        fname = path.join(dirname, "FSPS_Chabrier_neb_compact.pickle")
        lam_main = pickle.load(open(path.join(dirname, "lam_main.pickle"), "rb"))
        return cls(fname, lam_main)


    def reshape_sfh(self, sfh):
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

