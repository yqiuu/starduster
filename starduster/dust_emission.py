from .modules import PlankianMixture, create_mlp

import torch
from torch import nn


class AttenuationFractionSub(nn.Module):
    def __init__(self, input_size, hidden_sizes, activations, dropout=0.):
        super().__init__()
        self.mlp = create_mlp(input_size, hidden_sizes, activations)
        if dropout > 0:
            self.mlp.add_module('dropout', nn.Dropout(dropout))


    def forward(self, x_in):
        return torch.sum(self.mlp(x_in[0])*x_in[1], dim=1)


class AttenuationFraction(nn.Module):
    def __init__(self, lookup, frac_disk, frac_bulge, ):
        super().__init__()
        self.lookup = lookup
        self.frac_disk = frac_disk
        self.frac_bulge = frac_bulge


    def forward(self, x_in):
        x, lum_disk, lum_bulge = x_in
        b2t = .5*(x[:, self.lookup['b_o_t']] + 1) # Convert to range [0, 1]
        f_disk = self.frac_disk((x[:, self.lookup['frac_disk_inds']], lum_disk))
        f_bulge = self.frac_bulge((x[:, self.lookup['frac_bulge_inds']], lum_bulge))
        frac = f_disk*(1 - b2t) + f_bulge*b2t
        return frac


class EmissionDistribution(nn.Module):
    def __init__(self,
        input_size, output_size, hidden_sizes, activations,
        n_mix, lam, w_line, dx, eps=1e-20
    ):
        super().__init__()
        self.mlp = create_mlp(input_size, hidden_sizes, activations)
        self.base = PlankianMixture(hidden_sizes[-1], n_mix, lam)
        self.trans1 = TransferScalar(hidden_sizes[-1], output_size, dx)
        self.trans2 = TransferVector(hidden_sizes[-1], output_size, w_line, dx)
        self.dx = dx
        self.eps = eps


    def forward_(self, x):
        z = self.mlp(x)
        y = self.base(z)
        y0 = y/torch.trapz(y, dx=self.dx)[:, None] + self.eps
        y1 = y0 + self.trans1(z, y0)
        y2 = y1 + self.trans2(z, y1)
        return y0, y1, y2
    

    def forward(self, x):
        y0, y1, y2 = self.forward_(x)
        return y2


class TransferScalar(nn.Module):
    def __init__(self, input_size, output_size, dx):
        super().__init__()
        self.lin_neg = nn.Linear(input_size, output_size)
        self.dx = dx
        
        
    def forward(self, x, budget):
        z_neg = torch.sigmoid(self.lin_neg(x))*budget
        y = torch.trapz(z_neg, dx=self.dx)[:, None]*budget - z_neg
        return y


class TransferVector(nn.Module):
    def __init__(self, input_size, output_size, weights, dx):
        super().__init__()
        self.lin_neg = nn.Linear(input_size, output_size)
        self.lin_pos = nn.Linear(input_size, output_size)
        self.weights = weights
        self.dx = dx


    def forward(self, x, budget):
        z_pos = torch.exp(self.lin_pos(x))*self.weights*budget
        z_pos = z_pos/torch.trapz(z_pos, dx=self.dx)[:, None]
        z_neg = torch.sigmoid(self.lin_neg(x))*budget
        y = torch.trapz(z_neg, dx=self.dx)[:, None]*z_pos - z_neg
        return y

