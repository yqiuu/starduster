from .modules import Monotonic, Unimodal, Smooth, create_mlp

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


class EmissionDistribution(nn.Module):
    def __init__(self,
        forest_inds, input_size, output_size, hidden_sizes, activations,
        channels, kernel_size, dx, eps=1e-20
    ):
        super().__init__()
        self.mlp = create_mlp(input_size, hidden_sizes, activations)
        self.continuum = nn.Sequential(
            nn.Unflatten(1, (1, hidden_sizes[-1])),
            nn.Conv1d(
                in_channels=1,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1)//2,
                padding_mode='replicate',
                bias=False
            ),
            nn.BatchNorm1d(channels),
            nn.Upsample(output_size, mode='linear', align_corners=False),
            nn.Conv1d(
                in_channels=channels,
                out_channels=1,
                kernel_size=kernel_size,
                padding=(kernel_size - 1)//2,
                padding_mode='replicate',
                bias=False
            ),
            nn.Flatten(),
            nn.Softplus()
        )
        self.forest = nn.Sequential(
            nn.Linear(hidden_sizes[-1], forest_inds[1] - forest_inds[0]),
            nn.Softplus()
        )
        self.forest_slice = slice(*forest_inds)
        self.dx = dx
        self.eps = eps


    def components(self, x_in, combine=True):
        x = self.mlp(x_in)
        continuum = self.continuum(x)
        forest = torch.zeros_like(continuum)
        forest[:, self.forest_slice] = self.forest(x)
        y = continuum + forest
        norm = torch.trapz(y, dx=self.dx) + self.eps
        y /= norm[:, None]
        if combine:
            return y
        else:
            continuum /= norm[:, None]
            forest /= norm[:, None]
            return y, continuum, forest


    def forward(self, x_in):
        return self.components(x_in)
