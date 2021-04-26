from .utils import Evaluator

import torch
from torch import nn


__all__ = ["DAE", "EvaluatorDAE"]


class DAE(nn.Module):
    def __init__(self, input_size, n_hidden, n_latent):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(.5),
            nn.Linear(input_size, n_hidden),
            nn.Linear(n_hidden, n_latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.Linear(n_hidden, input_size)
        )
        self.n_latent = n_latent


    def forward(self, x):
        return self.decoder(self.encoder(x))


class EvaluatorDAE(Evaluator):
    def loss_func(self, *args):
        k_true, = args
        k_pred = self.model(k_true)
        delta = torch.norm(k_true - k_pred, float('inf'), dim=1)
        delta = delta/(1 - k_true[:, None, -1])
        return torch.mean(delta)

