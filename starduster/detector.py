import numpy as np
from astropy import units as U
from astropy import constants
from sedpy.observate import Filter as sedpy_Filter
import torch
from torch import nn


class Detector(nn.Module):
    """Apply filters to the input fluxes with unit conversion.

    Parameters
    ----------
    lam : tensor [micrometer]
        Wavelength of the input fluxes.
    """
    def __init__(self, lam):
        super().__init__()
        self.register_buffer('lam', lam)
        self.lam_base = lam
        self.configure()
        self._set_unit()


    def forward(self, l_target, return_ph, return_lum):
        ## The given flux should be in L_sol.
        if return_ph:
            return self.apply_filters(l_target)
        else:
            if return_lum:
                return l_target
            else:
                return l_target*self.lam*(self.unit_f_nu*self.dist_factor)


    def apply_filters(self, l_target):
        ## The given flux should be in L_sol.
        fluxes = torch.trapz(l_target[:, None, :]*self.trans_filter, self.lam)*self.dist_factor
        if self.ab_mag:
            return -2.5*torch.log10(fluxes) + 8.9
        else:
            return fluxes


    def configure(self, filters=None, redshift=0., distmod=0., ab_mag=True):
        """Configure the output mode.

        Parameters
        ----------
        filters : sedpy.observate.Filter
            Output filters.
        redshift : float
            Redshift.
        distmod : float
            Distance modulus.
        ab_mag : bool
            If ``True``, return AB magnitudes; otherwise return flux densities.
        """
        self._prepare_filters(filters, redshift)
        self.dist_factor = 10**(-.4*distmod)
        self.ab_mag = ab_mag
        self._config = {
            'filters': filters, 'redshift': redshift, 'distmod': distmod, 'ab_mag': ab_mag
        }


    def _prepare_filters(self, filters, redshift):
        lam = self.lam_base*(1 + redshift)
        self.register_buffer('lam', lam)
        if filters is not None:
            trans_filter = np.zeros([len(filters), len(lam)])
            lam_pivot = np.zeros(len(filters))
            for i_f, ftr in enumerate(filters):
                trans_filter[i_f], lam_pivot[i_f] = self._derive_filter_params(ftr, lam)
            self.register_buffer('trans_filter', torch.tensor(trans_filter, dtype=torch.float32))
            self.register_buffer('lam_pivot', torch.tensor(lam_pivot, dtype=torch.float32))


    def _derive_filter_params(self, ftr, lam):
        ## The given flux and wavelength should be in L_sol and micrometer
        ## respectively.
        unit_lam = U.angstrom.to(U.micrometer)
        if isinstance(ftr, sedpy_Filter):
            lam_0 = ftr.wavelength*unit_lam
            trans_0 = ftr.transmission
        else:
            lam_0, trans_0 = ftr
            lam_0 = lam_0*unit_lam
        # Compute transmission
        trans = np.interp(lam, lam_0, trans_0, left=0., right=0.)
        trans = trans/np.trapz(trans/lam, lam)*self.unit_jansky
        # Compute pivot wavelength
        lam_pivot = np.sqrt(np.trapz(lam_0*trans_0, lam_0)/np.trapz(trans_0/lam_0, lam_0))
        return trans, lam_pivot


    def _set_unit(self):
        unit_f_nu = U.solLum/(4*np.pi*(10*U.parsec)**2)*U.micrometer/constants.c
        self.unit_f_nu = unit_f_nu.to(U.jansky).value
        self.unit_jansky = self.unit_f_nu


    def _get_config(self):
        return self._config

