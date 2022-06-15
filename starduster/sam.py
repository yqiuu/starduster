from bisect import bisect_left

import numpy as np
import torch


__all__ = ["SemiAnalyticConventer"]


class SemiAnalyticConventer:
    """A class to convert properties from a semi-analytic model into parameters
    that can be accepted by the given SED model.

    Parameters
    ----------
    sed_model : MultiwavelengthSED
        The target SED model.
    age_bins : array [yr]
        Stellar age bins.
    """
    def __init__(self, sed_model, age_bins):
        self.helper = sed_model.helper
        self.lib_ssp = sed_model.lib_ssp
        self._tau_matrix = self._create_tau_matrix(age_bins)


    def __call__(self,
        theta, m_dust, r_dust, r_disk, r_bulge,
        sfh_mass_disk, sfh_metal_mass_disk, sfh_mass_bulge, sfh_metal_mass_bulge
    ):
        """Convert the input properties into parameters that can be accepted by
        the given SED model.

        Parameters
        ----------
        theta : array [deg]
            Inlincation angel.
        m_dust : array [M_sol]
            Dust mass.
        r_dust : array [kpc]
            Dust disk radius.
        r_disk : array [kpc]
            Stellar disk radius.
        r_bulge : array [kpc]
            Stellar bulge radius.
        sfh_mass_disk : array [M_sol]
            Gross stellar mass of the stellar disk in each age bin.
        sfh_metal_mass_disk : array [M_sol]
            Metal mass of the stellar disk in each age bin.
        sfh_mass_bulge : array [M_sol]
            Gross stellar mass of the stellar bulge in each age bin.
        sfh_metal_mass_bulge : array [M_sol]
            Metal mass of the stellar bulge in each age bin.

        Returns
        -------
        gp : tensor
            Galaxy parameters.
        sfh_disk : tensor
            Star formation history of the disk component.
        sfh_bulge : tensor
            Star formation history of the bulge component.
        """
        den_dust = m_dust/(2*np.pi*r_dust*r_dust)
        r_dust_to_rd = r_dust/r_disk
        sfh_disk, l_norm_disk = self._derive_sfh(sfh_mass_disk, sfh_metal_mass_disk)
        sfh_bulge, l_norm_bulge = self._derive_sfh(sfh_mass_bulge, sfh_metal_mass_bulge)
        l_norm = l_norm_disk + l_norm_bulge
        b_to_t = l_norm_bulge/l_norm
        gp_0 = np.vstack([theta, den_dust, r_dust_to_rd, r_disk, r_bulge, l_norm, b_to_t]).T
        gp = torch.as_tensor(self.helper.transform_all(gp_0, lib=np), dtype=torch.float32)
        return gp, sfh_disk, sfh_bulge


    def _create_tau_matrix(self, age_bins):
        tau_edges = self.lib_ssp.tau_edges.numpy()
        n_edge = len(tau_edges)
        d_tau_base = np.diff(tau_edges)

        matrix = np.zeros([len(age_bins) - 1, len(tau_edges) - 1], dtype=np.float32)
        for i_step in range(len(age_bins) - 1):
            t_lower = age_bins[i_step]
            t_upper = age_bins[i_step + 1]
            dt = t_upper - t_lower
            assert dt > 0, "Age bins must be strictly increasing."
            matrix_sub = np.zeros(len(d_tau_base))
            idx_lower = bisect_left(tau_edges, t_lower)
            idx_upper = bisect_left(tau_edges, t_upper)
            if idx_lower == 0:
                if idx_upper == 0:
                    raise ValueError("An age bin is entirely beyond the youngest age.")
                else:
                    idx_lower = 1
            if idx_upper == n_edge:
                if idx_lower == n_edge:
                    raise ValueError("An age bin is entirely beyond the oldest age.")
                else:
                    idx_upper = n_edge - 1
            if idx_upper == idx_lower:
                # Distribute the mass into one bin
                matrix_sub[idx_lower - 1] = 1.
            elif idx_upper > idx_lower:
                # Distibute the mass into two bins according to the time fraction
                d_tau = np.zeros(len(d_tau_base))
                d_tau[idx_lower : idx_upper-1] = d_tau_base[idx_lower : idx_upper-1]
                d_tau[idx_lower - 1] = (tau_edges[idx_lower] - t_lower)
                d_tau[idx_upper - 1] = (t_upper - tau_edges[idx_upper - 1])
                matrix_sub = d_tau/dt
            else:
                ValueError("Something wrong with the age bins.")
            matrix[i_step] = matrix_sub
        return matrix


    def _derive_sfh(self, sfh_mass, sfh_metal_mass):
        sfh_mass = np.matmul(sfh_mass, self._tau_matrix)
        sfh_metal_mass = np.matmul(sfh_metal_mass, self._tau_matrix)
        sfh_met = self._derive_met(sfh_mass, sfh_metal_mass)
        sfh_mass = sfh_mass[:, None, :]*self._interpolate_met(sfh_met)
        sfh_mass = torch.as_tensor(sfh_mass, dtype=torch.float32)
        sfh_frac, l_norm = self.lib_ssp.mass_to_light(sfh_mass)
        sfh_frac = sfh_frac.flatten(start_dim=1)
        return sfh_frac, l_norm


    def _derive_met(self, sfh_mass, sfh_metal_mass):
        cond = sfh_mass > 0.
        sfh_met = np.zeros_like(sfh_mass)
        sfh_met[cond] = sfh_metal_mass[cond]/sfh_mass[cond]
        return sfh_met


    def _interpolate_met(self, sfh_met):
        met_centres = self.lib_ssp.met.numpy()
        weights = np.zeros([len(sfh_met), len(met_centres), sfh_met.shape[-1]], dtype=np.float32)
        met = np.copy(sfh_met) # Make a copy
        met[met < met_centres[0]] = met_centres[0]
        met[met > met_centres[-1]] = met_centres[-1]
        inds = np.searchsorted(met_centres, met)
        inds[inds == 0] = 1
        met_lower = met_centres[inds - 1]
        met_upper = met_centres[inds]
        weights_lower = (met_upper - met)/(met_upper - met_lower)
        weights_upper = 1 - weights_lower
        for i_tau in range(weights.shape[-1]):
            weights[range(len(met)), inds[:, i_tau] - 1, i_tau] = weights_lower[:, i_tau]
            weights[range(len(met)), inds[:, i_tau], i_tau] = weights_upper[:, i_tau]
        return weights

