from .modules import Monotonic, Unimodal, Smooth, create_mlp

import torch
from torch import nn


class AttenuationCurve(nn.Module):
    def __init__(self,
        input_size, output_size, hidden_sizes, activations, bump_inds, trough_ind,
        baseline_kernel_size=1, bump_kernel_size=1
    ):
        super().__init__()
        self.mlp = create_mlp(input_size, hidden_sizes, activations)
        self.norm = nn.Sequential(
            nn.Linear(hidden_sizes[-1], 1),
            nn.Softplus()
        )
        self.baseline = nn.Sequential(
            nn.Linear(hidden_sizes[-1], output_size + baseline_kernel_size - 1),
            Monotonic(increase=False),
            Smooth(baseline_kernel_size)
        )
        for i_bump, (idx_b, idx_e) in enumerate(bump_inds):
            seq = nn.Sequential(
                Unimodal(hidden_sizes[-1], idx_e - idx_b + bump_kernel_size - 1),
                Smooth(bump_kernel_size)
            )
            setattr(self, 'bump{}'.format(i_bump), seq)
        self.bump_inds = bump_inds
        self.trough = nn.Sequential(
            Unimodal(hidden_sizes[-1], trough_ind[1] - trough_ind[0] + bump_kernel_size - 1),
            Smooth(bump_kernel_size)
        )
        self.trough_ind = trough_ind


    def forward(self, x_in):
        x = self.mlp(x_in)
        y = self.baseline(x)
        for i_bump, (idx_b, idx_e) in enumerate(self.bump_inds):
            y[:, idx_b:idx_e] += getattr(self, 'bump{}'.format(i_bump))(x)
        y[:, slice(*self.trough_ind)] -= self.trough(x)
        y = y*self.norm(x)/torch.mean(y, dim=1)[:, None]
        return y


class DustAttenuation(nn.Module):
    def __init__(self, helper, curve_disk, curve_bulge, l_ssp):
        super().__init__()
        self.helper = helper
        self.curve_disk = curve_disk
        self.curve_bulge = curve_bulge
        self.l_ssp = l_ssp
        #
        self._b2t_name = 'b_to_t'


    def forward(self, params, sfh_disk, sfh_bulge):
        b2t = self.helper.get_recover(params, self._b2t_name, torch)[:, None]
        p_disk = self.helper.get_item(params, 'curve_disk_inds')
        p_bulge = self.helper.get_item(params, 'curve_bulge_inds')
        l_disk = torch.matmul(sfh_disk, self.l_ssp)
        l_bulge = torch.matmul(sfh_bulge, self.l_ssp)
        trans_disk = self.helper.set_item(
            torch.ones_like(l_disk), 'slice_lam_da', 10**(-.4*self.curve_disk(p_disk))
        )
        trans_bulge = self.helper.set_item(
            torch.ones_like(l_bulge), 'slice_lam_da', 10**(-.4*self.curve_bulge(p_bulge))
        )
        l_main = l_disk*trans_disk*(1 - b2t) + l_bulge*trans_bulge*b2t
        return l_main

