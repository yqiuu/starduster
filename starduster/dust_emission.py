from .modules import PlankianMixture, create_mlp, kld_trapz, kld_binary, reduce_loss

import torch
from torch import nn
from torch.nn import functional as F


class AttenuationFractionSub(nn.Module):
    def __init__(self, input_size, hidden_sizes, activations, dropout=0., sed=False):
        super().__init__()
        self.mlp = create_mlp(input_size, hidden_sizes, activations)
        if dropout > 0:
            self.mlp.add_module('dropout', nn.Dropout(dropout))
        self.sed = sed


    def forward(self, x_in):
        if self.sed:
            return torch.sum(self.mlp(x_in[0][:, None, :])*x_in[1], dim=-1)
        else:
            return torch.sum(self.mlp(x_in[0])*x_in[1], dim=-1)


class DustEmission(nn.Module):
    def __init__(self, helper, distri, frac_disk, frac_bulge, L_ssp=None):
        super().__init__()
        self.helper = helper
        self.distri = distri
        self.frac_disk = frac_disk
        self.frac_bulge = frac_bulge
        if L_ssp is None:
            self.L_ssp = L_ssp
        else:
            frac_disk.sed = True
            frac_bulge.sed = True
            self.register_buffer('L_ssp', L_ssp)


    @classmethod
    def from_args(
        cls, helper, kwargs_distri, kwargs_frac_disk, kwargs_frac_bulge, L_ssp=None
    ):
        distri = EmissionDistribution(**kwargs_distri)
        frac_disk = AttenuationFractionSub(**kwargs_frac_disk)
        frac_bulge = AttenuationFractionSub(**kwargs_frac_bulge)
        return cls(helper, distri, frac_disk, frac_bulge, L_ssp=L_ssp)


    @classmethod
    def from_checkpoint(cls, fname, L_ssp=None, no_dropout=True):
        checkpoint = torch.load(fname)
        if L_ssp is not None:
            checkpoint['model_state_dict']['L_ssp'] = L_ssp
        if no_dropout:
            checkpoint['params'][2]['dropout'] = 0.
            checkpoint['params'][3]['dropout'] = 0.
        model = cls.from_args(*checkpoint['params'], L_ssp=L_ssp)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


    def forward(self, x_in):
        x, lum_disk, lum_bulge = x_in
        frac, frac_disk, frac_bulge = self._fraction(x, lum_disk, lum_bulge)
        x = torch.cat([x, frac_disk[:, None], frac_bulge[:, None]], dim=1)
        l_dust = self.distri(x)
        return l_dust, frac


    def _fraction(self, x, y_disk, y_bulge):
        b2t = self.helper.recover(x, 'b_o_t')
        x_disk = self.helper.get_item(x, 'frac_disk_inds')
        x_bulge = self.helper.get_item(x, 'frac_bulge_inds')
        if self.L_ssp is None:
            frac_disk = self.frac_disk((x_disk, y_disk))
            frac_bulge = self.frac_bulge((x_bulge, y_bulge))
        else:
            frac_disk = torch.sum(y_disk*self.frac_disk((x_disk, self.L_ssp)), dim=-1)
            frac_bulge = torch.sum(y_bulge*self.frac_bulge((x_bulge, self.L_ssp)), dim=-1)
        frac = (frac_disk*(1 - b2t) + frac_bulge*b2t)[:, None]
        return frac, frac_disk, frac_bulge


class EmissionDistribution(nn.Module):
    def __init__(self,
        input_size, output_size, hidden_sizes, activations, n_mix, lam, dx
    ):
        super().__init__()
        self.mlp = create_mlp(input_size, hidden_sizes, activations)
        self.base = PlankianMixture(hidden_sizes[-1], n_mix, lam)
        self.transfer = Transfer(hidden_sizes[-1], output_size, dx)
        self.dx = dx


    def forward(self, x):
        z = self.mlp(x)
        y = self.base(z)
        y0 = y/torch.trapz(y, dx=self.dx)[:, None]
        y1 = y0 + self.transfer(z, y0)
        return y1


class Transfer(nn.Module):
    def __init__(self, input_size, output_size, dx):
        super().__init__()
        self.lin_neg = nn.Linear(input_size, output_size)
        self.lin_pos = nn.Linear(input_size, output_size)
        self.dx = dx


    def forward(self, x, budget):
        z_pos = F.softplus(self.lin_pos(x))
        z_pos = z_pos/torch.trapz(z_pos, dx=self.dx)[:, None]
        z_neg = torch.sigmoid(self.lin_neg(x))*budget
        y = torch.trapz(z_neg, dx=self.dx)[:, None]*z_pos - z_neg
        return y


class LossDE(nn.Module):
    def __init__(self, dx, eps_trapz=1e-10, eps_binary=1e-6, reduction='mean'):
        super().__init__()
        self.dx = dx
        self.eps_trapz = eps_trapz
        self.eps_binary = eps_binary
        self.reduction = reduction


    def forward(self, y_pred, y_true):
        distri_pred, frac_pred = y_pred
        distri_true, frac_true = y_true
        l_distri = reduce_loss(
            kld_trapz(distri_pred, distri_true, self.dx, self.eps_trapz), self.reduction
        )
        l_frac = reduce_loss(
            kld_binary(frac_pred, frac_true, self.eps_binary), self.reduction
        )
        return l_distri + l_frac, l_distri, l_frac

