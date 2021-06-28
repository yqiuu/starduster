from .evaluator import Evaluator

import torch
from torch import nn


__all__ = ["AE", "EvaluatorAE"]


class AE(nn.Module):
    def __init__(self, input_size, n_hidden, n_latent):
        super().__init__()
        self.encoder = nn.Sequential(
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


class EvaluatorAE(Evaluator):
    def loss_func(self, *args):
        k_true, = args
        k_pred = self.model(k_true)
        delta = torch.norm(k_true - k_pred, float('inf'), dim=1) \
            / (1 - k_true[:, None, -1])
        return torch.mean(delta)

