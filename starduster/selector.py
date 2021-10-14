from .utils import accept_reject
from .modules import create_mlp
from .helper import Helper

import torch
from torch import nn


class Selector(nn.Module):
    def __init__(self, helper, input_size, hidden_sizes, activations):
        super().__init__()
        assert hidden_sizes[-1] == 1
        if isinstance(helper, Helper):
            self.helper = helper
        else:
            self.helper = Helper(*helper)
        self.mlp = create_mlp(input_size, hidden_sizes, activations)


    def forward(self, gp):
        return self.mlp(gp)


    def select(self, gp):
        return torch.ravel(torch.sigmoid(self.forward(gp)) > .5)


    def sample(self, n_samp, sampler=None):
        n_col = len(self.helper.header)
        params_accept = torch.zeros([0, n_col])
        while len(params_accept) < n_samp:
            if sampler is None:
                params = 2*torch.rand(2*n_samp, n_col) - 1.
            else:
                params = sampler(2*n_samp)
            params = params[self.select(params)]
            params_accept = torch.vstack([params_accept, params])
        params_accept = self.helper.recover_all(params_accept[:n_samp], torch)
        return params_accept


def sample_from_selector(
    n_samp, selector_disk=None, selector_bulge=None, sampler=None, max_iter=10000
):
    if selector_disk is None and selector_bulge is None:
        raise ValueError("No selector is provided.")
    elif selector_disk is not None:
        helper = selector_disk.helper
    else:
        helper = selector_bulge.helper

    if sampler is None:
        if selector_bulge is not None:
            slice_col = helper.lookup['curve_bulge_inds']
        else:
            slice_col = helper.lookup['curve_disk_inds']
        n_col = slice_col.stop - slice_col.start
        sampler = lambda n_samp: 2*torch.rand(n_samp, n_col) - 1.
    else:
        n_col = sampler(0).size(1)

    def condition(params):
        cond = torch.full((params.size(0),), True)
        if selector_disk is not None:
            cond &= selector_disk.select(helper.get_item(params, 'curve_disk_inds'))
        if selector_bulge is not None:
            cond &= selector_bulge.select(helper.get_item(params, 'curve_bulge_inds'))
        return cond

    return accept_reject(n_samp, n_col, sampler, condition, max_iter)

