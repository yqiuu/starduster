from .utils import Evaluator

import torch
from torch import nn


__all__ = ["DualSequential", "Evaluator_Curve"]


class DualSequential(nn.Module):
    def __init__(self, n_shape, layer_sizes):
        super().__init__()
        layers = []
        for i, k in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.extend([nn.Linear(i, k), nn.Tanh()])
        self.seq = nn.Sequential(*layers)
        self.lin_shape = nn.Linear(layer_sizes[-1], n_shape)
        self.lin_norm = nn.Linear(layer_sizes[-1], 1)


    def forward(self, x):
        x = self.seq(x)
        z = self.lin_shape(x)
        norm = nn.functional.softplus(self.lin_norm(x))
        return z, norm


class Evaluator_Curve(Evaluator):
    def __init__(self, model, opt, auto_encoder, inds_include=None):
        super().__init__(model, opt, ("loss",))
        self.auto_encoder = auto_encoder
        self.set_adapter(inds_include)


    def set_adapter(self, inds_include):
        n_latent = self.auto_encoder.n_latent
        if inds_include is None:
            inds_include = np.arange(n_latent)
        adapter = torch.zeros([len(inds_include), n_latent])
        for i, j in enumerate(inds_include):
            adapter[i, j] = 1.
        self.adapter = adapter


    def loss_func(self, x, y):
        k_true, norm_true = y
        z, norm_pred = self.model(x)
        k_pred = self.auto_encoder.decoder(torch.matmul(z, self.adapter))
        delta = norm_pred*k_pred - norm_true*k_true
        delta = torch.norm(delta, float('inf'), dim=1)
        delta = delta/(1 - k_true[:, None, -1])
        return torch.mean(delta)

