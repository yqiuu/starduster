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
            Unimodal(hidden_sizes[-1], trough_ind[1] - trough_ind[0] + baseline_kernel_size - 1),
            Smooth(baseline_kernel_size)
        )
        self.trough_ind = trough_ind
        #
        self._norm_fix = .1


    def forward(self, x_in):
        x = self.mlp(x_in)
        y = self.baseline(x)
        for i_bump, (idx_b, idx_e) in enumerate(self.bump_inds):
            y[:, idx_b:idx_e] += getattr(self, 'bump{}'.format(i_bump))(x)
        y[:, slice(*self.trough_ind)] -= self.trough(x)
        y = y*self.norm(x)/(torch.mean(y, dim=1)[:, None] + self._norm_fix)
        return y


class DustAttenuation(nn.Module):
    def __init__(self, helper, curve_disk, curve_bulge, l_ssp, interp=None):
        super().__init__()
        self.helper = helper
        self.curve_disk = curve_disk
        self.curve_bulge = curve_bulge
        self.register_buffer('l_ssp', l_ssp, persistent=False)
        self.interp = interp
        #
        self._b2t_name = 'b_to_t'


    def forward(self, params, sfh_disk, sfh_bulge):
        b2t = self.helper.get_recover(params, self._b2t_name, torch)[:, None]
        l_disk = torch.matmul(sfh_disk, self.l_ssp)
        l_disk = self.apply_transmission('disk', l_disk, params)
        l_bulge = torch.matmul(sfh_bulge, self.l_ssp)
        l_bulge = self.apply_transmission('bulge', l_bulge, params)
        l_main = l_disk*(1 - b2t) + l_bulge*b2t
        return l_main


    def apply_transmission(self, target, l_target, params):
        params_curve = self.helper.get_item(params, f'curve_{target}_inds')
        trans = 10**(-.4*getattr(self, f'curve_{target}')(params_curve))
        if self.interp is None:
            trans = self.helper.set_item(torch.ones_like(l_target), 'slice_lam_da', trans)
        else:
            trans = self.interp(trans)
        return trans*l_target

