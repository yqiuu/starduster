from .modules import PlankianMixture, create_mlp

import torch
from torch import nn


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

