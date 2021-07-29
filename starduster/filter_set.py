import numpy as np
import torch
from torch import nn

class FilterSet(nn.Module):
    """Apply filters to the input fluxes.
    
    Parameters
    ----------
    filters : array
        an array of pyphot filter instances.
    lam : array [AA]
        Wavelength of the input fluxes.
    """
    def __init__(self, filters, lam):
        super().__init__()
        trans_filter = np.zeros([len(filters), len(lam)])
        for i_f, f in enumerate(filters):
            lam_f = f.wavelength.value
            trans_f = f.transmit/np.trapz(lam_f*f.transmit, lam_f)
            trans_filter[i_f] = np.interp(lam, lam_f, trans_f)
        self.register_buffer('trans_filter', torch.tensor(trans_filter, dtype=torch.float32))
        self.register_buffer('lam', lam)
        
        
    def forward(self, l_target):
        """
        Parameters
        ----------
        l_target : tensor [?]
            Generalized flux density (nu*f_nu).
        """
        return torch.trapz(l_target[:, None, :]*self.trans_filter, self.lam)

