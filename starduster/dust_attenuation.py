from .utils import create_mlp, reduce_loss

import torch
from torch import nn
from torch.nn import functional as F


class Monotonic(nn.Module):
    def __init__(self, increase=True):
        super().__init__()
        self.increase = increase


    def forward(self, x_in):
        x = F.softplus(x_in)
        x = torch.cumsum(x, dim=1)/x.size(1)
        if self.increase:
            return x - x[:, None, 0]
        else:
            return -x + x[:, None, -1]


class Unimodal(nn.Module):
    def __init__(self):
        super().__init__()
        self.increase = Monotonic(increase=True)
        self.decrease = Monotonic(increase=False)


    def forward(self, x_in):
        return self.increase(x_in)*self.decrease(x_in)


class Smooth(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size


    def forward(self, x_in):
        x = F.avg_pool1d(x_in[:, None, :], self.kernel_size, 1)
        return x[:, 0, :]


class AttenuationCurve(nn.Module):
    def __init__(self,
        input_size, output_size, hidden_sizes, activations, bump_inds,
        baseline_kernel_size=1, bump_kernel_size=1
    ):
        super().__init__()
        self.mlp = create_mlp(input_size, hidden_sizes, activations)
        self.baseline = nn.Sequential(
            nn.Linear(hidden_sizes[-1], output_size + baseline_kernel_size - 1),
            Monotonic(increase=False),
            Smooth(baseline_kernel_size)
        )
        for i_bump, (idx_b, idx_e) in enumerate(bump_inds):
            seq = nn.Sequential(
                nn.Linear(hidden_sizes[-1], idx_e - idx_b + bump_kernel_size - 1),
                Unimodal(),
                Smooth(bump_kernel_size)
            )
            setattr(self, 'bump{}'.format(i_bump), seq)
        self.bump_inds = bump_inds


    def forward(self, x_in):
        x = self.mlp(x_in)
        y = self.baseline(x)
        for i_bump, (idx_b, idx_e) in enumerate(self.bump_inds):
            y[:, idx_b:idx_e] += getattr(self, 'bump{}'.format(i_bump))(x)
        return y

