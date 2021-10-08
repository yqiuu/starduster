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
        l_ssp = interp_arr(np.log(lam), log_lam_ssp, l_ssp_raw)
        # Save attributes
        self.tau = torch.tensor(lib_ssp['tau'], dtype=torch.float32)
        self.met = torch.tensor(lib_ssp['met'], dtype=torch.float32)
        self.lam = torch.tensor(lam, dtype=torch.float32)
        self.l_ssp = torch.tensor(l_ssp, dtype=torch.float32)
        self.L_ssp = torch.tensor(L_ssp, dtype=torch.float32)
        self.norm = torch.tensor(lib_ssp['norm'], dtype=torch.float32)
        self.tau_edges = torch.tensor(lib_ssp['tau_edges'], dtype=torch.float32)
        self.d_tau = torch.diff(self.tau_edges)


    def reshape_sfh(self, sfh):
        return torch.atleast_2d(sfh).reshape((-1, *self.sfh_shape))


    def sum_over_age(self, sfh):
        return self.reshape_sfh(sfh).sum(dim=self.dim_age)


    def sum_over_met(self, sfh):
        return self.reshape_sfh(sfh).sum(dim=self.dim_met)


def load_builtin_library():
    dirname = path.join(path.dirname(path.abspath(__file__)), "database")
    fname = path.join(dirname, "FSPS_Chabrier_neb_compact.pickle")
    lam_main = pickle.load(open(path.join(dirname, "lam_main.pickle"), "rb"))
    return SSPLibrary(fname, lam_main)

