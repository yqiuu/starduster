from .evaluator import Evaluator

import numpy as np
import torch
from torch import nn


__all__ = ["ExtinctionCurve", "EvaluatorCurve"]


def create_sequential(layer_sizes):
    layers = []
    for i, k in zip(layer_sizes[:-1], layer_sizes[1:]):
        layers.extend([nn.Linear(i, k), nn.Tanh()])
    return nn.Sequential(*layers)


class ResNet(nn.Module):
    def __init__(self, layer_sizes, n_res):
        assert layer_sizes[0] == layer_sizes[-1]

        super().__init__()
        for i_res in range(n_res):
            setattr(self, f'mlp{i_res}', create_sequential(layer_sizes))
        self.n_res = n_res


    def forward(self, x):
        for i_res in range(self.n_res):
            x = x + getattr(self, f'mlp{i_res}')(x)
        return x


class ExtinctionCurve(nn.Module):
    def __init__(self,
        auto_encoder, layer_sizes, res_layer_sizes, n_res
    ):
        super().__init__()
        self._auto_encoder = (auto_encoder,)
        self.resnet = ResNet(res_layer_sizes, n_res)
        self.mlp = create_sequential(layer_sizes)
        self.lin_shape = nn.Linear(layer_sizes[-1], auto_encoder.n_latent)
        self.lin_norm = nn.Linear(layer_sizes[-1], 1)

    
    @property
    def auto_encoder(self):
        return self._auto_encoder[0]


    def forward(self, x):
        x = self.resnet(x)
        x = self.mlp(x)
        z = self.lin_shape(x)
        norm = nn.functional.softplus(self.lin_norm(x))
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

