from .utils import Evaluator

import numpy as np
import torch
from torch import nn


__all__ = ["ExtinctionCurve", "EvaluatorCurve"]


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


class ExtinctionCurve(nn.Module):
    def __init__(self, auto_encoder, layer_sizes, inds_include=None):
        super().__init__()
        if inds_include is None:
            inds_include = np.arange(auto_encoder.n_latent)
        self._auto_encoder = (auto_encoder,)
        self.set_adapter(inds_include)
        self.dual_seq = DualSequential(len(inds_include), layer_sizes)

    
    @property
    def auto_encoder(self):
        return self._auto_encoder[0]


    def set_adapter(self, inds_include):
        adapter = torch.zeros([len(inds_include), self.auto_encoder.n_latent])
        for i, j in enumerate(inds_include):
            adapter[i, j] = 1.
        self.adapter = adapter

    
    def forward(self, x):
        z, norm = self.dual_seq(x)
        k = self.auto_encoder.decoder(torch.matmul(z, self.adapter))
        return k*norm


class EvaluatorCurve(Evaluator):
    def __init__(self, model, opt):
        super().__init__(model, opt, ("loss",))


    def loss_func(self, *args):
        x, k_true, norm_true = args
        kn_pred = self.model(x)
        delta = torch.norm(kn_pred - norm_true*k_true, float('inf'), dim=1) \
            / (1 - k_true[:, None, -1])
        return torch.mean(delta)

