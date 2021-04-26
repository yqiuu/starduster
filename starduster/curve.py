from .utils import Evaluator

import numpy as np
import torch
from torch import nn


__all__ = ["ExtinctionCurve", "EvaluatorCurve"]


class DualSequential(nn.Module):
    def __init__(self, n_out, layer_sizes):
        super().__init__()
        layers = []
        for i, k in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.extend([nn.Linear(i, k), nn.Tanh()])
        self.seq = nn.Sequential(*layers)
        self.lin_shape = nn.Linear(layer_sizes[-1], n_out)
        self.lin_norm = nn.Linear(layer_sizes[-1], 1)


    def forward(self, x):
        x = self.seq(x)
        z = self.lin_shape(x)
        norm = nn.functional.softplus(self.lin_norm(x))
        return z, norm


class ExtinctionCurve(nn.Module):
    def __init__(self, auto_encoder, layer_sizes):
        super().__init__()
        self._auto_encoder = (auto_encoder,)
        self.dual_seq = DualSequential(auto_encoder.n_latent, layer_sizes)

    
    @property
    def auto_encoder(self):
        return self._auto_encoder[0]


    def forward(self, x):
        z, norm = self.dual_seq(x)
        k = self.auto_encoder.decoder(z)
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

