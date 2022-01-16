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
            pset_gp=GalaxyParameter(),
            pset_sfh_disk=VanillaGrid(),
            pset_sfh_bulge=VanillaGrid(),
            flat_input=False,
            check_sfh_norm=True,
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
        for name, pset in self.get_config().items():
            if name.startswith('pset'):
                pset.enable(self.helper, self.lib_ssp)
                free_shape.append(pset.input_size)
                param_names.extend(pset.param_names)
                bounds.append(pset.bounds)
        self.free_shape = free_shape
        self.input_size = sum(self.free_shape)
        self.param_names = param_names
        self.bounds = np.vstack(bounds)


    def forward(self, *args, check_bounds=False):
        if self.flat_input:
            params = torch.as_tensor(args[0], dtype=torch.float32, device=self.device)
            params = torch.atleast_2d(params)
            gp, sfh_disk, sfh_bulge = torch.split(params, self.free_shape, dim=-1)
        else:
            gp, sfh_disk, sfh_bulge = args

        if check_bounds:
            # Check if all input parameters are within the bounds
            is_out = torch.full((gp.size(0),), False, device=self.device)
            for val, pset in zip(
                [gp, sfh_disk, sfh_bulge],
                [self.pset_gp, self.pset_sfh_disk, self.pset_sfh_bulge]
            ):
                is_out |= pset.check_bounds(val)
            # Apply the parameterisations
            gp, sfh_disk, sfh_bulge = self._apply_pset(gp, sfh_disk, sfh_bulge)
            # Check if all parameters are in the effective region
            # -Assume helper of selector_disk and selector_bulge are the same
            helper = self.selector_disk.helper
            is_out |= ~self.selector_disk.select(helper.get_item(gp, 'curve_disk_inds'))
            is_out |= ~self.selector_bulge.select(helper.get_item(gp, 'curve_bulge_inds'))
            #
            return gp, sfh_disk, sfh_bulge, is_out
        else:
            return self._apply_pset(gp, sfh_disk, sfh_bulge)


    @property
    def device(self):
        return self._device.device


    def _apply_pset(self, gp, sfh_disk, sfh_bulge):
        gp = self.pset_gp(gp)
        sfh_disk = self.pset_sfh_disk(sfh_disk)
        sfh_bulge = self.pset_sfh_bulge(sfh_bulge)

        if self.check_sfh_norm:
            msg = "Star formation history must be normalised to one."
            assert torch.allclose(sfh_disk.sum(dim=-1), torch.tensor(1.), atol=1e-5), msg
            assert torch.allclose(sfh_bulge.sum(dim=-1), torch.tensor(1.), atol=1e-5), msg

        return gp, sfh_disk, sfh_bulge

