from .modules import Monotonic, Unimodal, Smooth, create_mlp, PlankianMixture

import torch
from torch import nn


class AttenuationCurve(nn.Module):
    def __init__(self,
        input_size, output_size, hidden_sizes, activations, bump_inds,
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


    def forward(self, x_in):
        x = self.mlp(x_in)
        y = self.baseline(x)
        for i_bump, (idx_b, idx_e) in enumerate(self.bump_inds):
            y[:, idx_b:idx_e] += getattr(self, 'bump{}'.format(i_bump))(x)
        y = y*self.norm(x)/torch.mean(y, dim=1)[:, None]
        return y


class AttenuationFraction(nn.Module):
    def __init__(self, input_size, hidden_sizes, activations, dropout=0.):
        super().__init__()
        self.mlp = create_mlp(input_size, hidden_sizes, activations)
        if dropout > 0:
            self.mlp.add_module('dropout', nn.Dropout(dropout))


    def forward(self, x_in):
        return torch.sum(self.mlp(x_in[0])*x_in[1], dim=1)

