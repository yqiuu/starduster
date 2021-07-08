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


class EmissionDistribution(nn.Module):
    def __init__(self,
        input_size, output_size, hidden_sizes, activations, n_mix, x, dx, eps=1e-20
    ):
        super().__init__()
        self.mlp = create_mlp(input_size, hidden_sizes, activations)
        self.continuum = PlankianMixture(hidden_sizes[-1], n_mix, x)
        self.zero = TransitionLinear(hidden_sizes[-1], output_size, dx)
        self.res_net =ResNet(output_size, 128, dx)
        self.dx = dx
        self.eps = eps


    def forward_base(self, x):
        z = self.mlp(x)
        y = self.continuum(z)
        y = y/torch.trapz(y, dx=self.dx)[:, None] + self.eps
        y_fix = self.zero(z, y)
        return y, y_fix
    

    def forward(self, x):
        y, y_fix = self.forward_base(x)
        y = self.res_net(y + y_fix)
        return y


class TransitionLinear(nn.Module):
    def __init__(self, input_size, output_size, dx):
        super().__init__()
        self.seq_pos = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Softplus()
        )
        self.seq_neg = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Sigmoid()
        )
        self.dx = dx


    def forward(self, x, budget):
        z_pos = self.seq_pos(x)
        z_pos = z_pos/torch.trapz(z_pos, dx=self.dx)[:, None]
        z_neg = -self.seq_neg(x)*budget
        y = -torch.trapz(z_neg, dx=self.dx)[:, None]*z_pos + z_neg
        return y



class ResNet(nn.Module):
    def __init__(self, output_size, hidden_size, dx):
        super().__init__()
        self.lin = nn.Linear(output_size, hidden_size, bias=False)
        self.zero = TransitionLinear(hidden_size, output_size, dx)


    def forward(self, x):
        z = self.lin(x)
        y = self.zero(z, x) + x
        return y


