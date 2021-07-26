from .modules import PlankianMixture, create_mlp, kld_trapz, kld_binary, reduce_loss

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


class DustEmissionFit(nn.Module):
    def __init__(self, lookup, distri, frac_disk, frac_bulge):
        super().__init__()
        self.lookup = lookup
        self.distri = distri
        self.frac_disk = frac_disk
        self.frac_bulge = frac_bulge


    def _fraction(self, x, lum_disk, lum_bulge):
        b2t = .5*(x[:, self.lookup['b_o_t']] + 1) # Convert to range [0, 1]
        frac_disk = self.frac_disk((x[:, self.lookup['frac_disk_inds']], lum_disk))
        frac_bulge = self.frac_bulge((x[:, self.lookup['frac_bulge_inds']], lum_bulge))
        frac = (frac_disk*(1 - b2t) + frac_bulge*b2t)[:, None]
        return frac, frac_disk, frac_bulge


    def forward(self, x_in):
        x, lum_disk, lum_bulge = x_in
        frac, frac_disk, frac_bulge = self._fraction(x, lum_disk, lum_bulge)
        x = torch.cat([x, frac_disk[:, None], frac_bulge[:, None]], dim=1)
        l_dust = self.distri(x)
        return l_dust, frac


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
        self.register_buffer('weights', weights)
        self.dx = dx


    def forward(self, x, budget):
        z_pos = torch.exp(self.lin_pos(x))*self.weights*budget
        z_pos = z_pos/torch.trapz(z_pos, dx=self.dx)[:, None]
        z_neg = torch.sigmoid(self.lin_neg(x))*budget
        y = torch.trapz(z_neg, dx=self.dx)[:, None]*z_pos - z_neg
        return y


class LossDE(nn.Module):
    def __init__(self, dx, eps=1e-10, reduction='mean'):
        super().__init__()
        self.dx = dx
        self.eps = eps
        self.reduction = reduction


    def forward(self, y_pred, y_true):
        distri_pred, frac_pred = y_pred
        distri_true, frac_true = y_true
        l_distri = reduce_loss(
            kld_trapz(distri_pred, distri_true, self.dx, self.eps), self.reduction
        )
        l_frac = reduce_loss(
            kld_binary(frac_pred, frac_true, self.eps), self.reduction
        )
        return l_distri + l_frac, l_distri, l_frac

