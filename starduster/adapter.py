from .utils import Configurable
from .parametrization import GalaxyParameter, VanillaGrid

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Adapter(nn.Module, Configurable):
    """Apply different parametrisations to input parameters.

    Parameters
    ----------
    helper : Helper
        Helper of input parameters.
    lib_ssp : SSPLibrary
        A simple stellar population library.
    selector_disk : Selector
        Selector of the disk component.
    selector_bulge : Selector
        Selector of the bulge component.
    """
    def __init__(self, helper, lib_ssp, selector_disk=None, selector_bulge=None):
        self.helper = helper
        self.lib_ssp = lib_ssp
        nn.Module.__init__(self)
        Configurable.__init__(self,
            pn_gp=GalaxyParameter(),
            pn_sfh_disk=VanillaGrid(),
            pn_sfh_bulge=VanillaGrid(),
            share_pn='none',
            flat_input=False,
        )
        if selector_disk is not None:
            self.selector_disk = selector_disk
        if selector_bulge is not None:
            self.selector_bulge = selector_bulge
        self.register_buffer("_device", torch.tensor(0.), persistent=False)


    def update_config(self):
        free_shape = []
        param_names = []
        bounds = []
        for name, pn in self.get_config().items():
            if name.startswith('pn'):
                pn.enable(self.helper, self.lib_ssp)
                if self.share_pn == 'all' and name != 'pn_gp':
                    continue
                if self.share_pn == 'sfh' and name == 'pn_sfh_bulge':
                    continue
                free_shape.append(pn.input_size)
                param_names.extend(pn.param_names)
                bounds.append(pn.bounds)
        self.free_shape = free_shape
        self.input_size = sum(self.free_shape)
        self.param_names = param_names
        self.bounds = np.vstack(bounds)


    def forward(self, *args, check_selector=False):
        if self.share_pn == 'all':
            gp, sfh_disk, sfh_bulge = self.pn_gp(*args)
        else:
            if self.flat_input:
                params = torch.atleast_2d(args[0])
                if self.share_pn == 'sfh':
                    gp, sfh_disk = torch.split(params, self.free_shape, dim=-1)
                    sfh_bulge = sfh_disk
                else:
                    gp, sfh_disk, sfh_bulge = torch.split(params, self.free_shape, dim=-1)
            else:
                if self.share_pn == 'sfh':
                    gp, sfh_disk = args
                    sfh_bulge = sfh_disk
                else:
                    gp, sfh_disk, sfh_bulge = args
            gp = self.pn_gp(gp)
            sfh_disk = self.pn_sfh_disk(sfh_disk)
            sfh_bulge = self.pn_sfh_bulge(sfh_bulge)

        if check_selector:
            return gp, sfh_disk, sfh_bulge, self._eval_selector(gp)
        else:
            return gp, sfh_disk, sfh_bulge


    @property
    def device(self):
        return self._device.device


    def _eval_selector(self, gp):
        p_in_disk = self.selector_disk(self.helper.get_item(gp, 'curve_disk_inds'))
        p_in_bulge = self.selector_bulge(self.helper.get_item(gp, 'curve_bulge_inds'))
        return F.logsigmoid(p_in_disk), F.logsigmoid(p_in_bulge)

