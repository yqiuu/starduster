from .utils import Evaluator

import numpy as np
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_size, n_hidden, n_latent):
        super().__init__()
        self.lin1 = nn.Linear(input_size, n_hidden)
        self.lin_mu = nn.Linear(n_hidden, n_latent)
        self.lin_var = nn.Linear(n_hidden, n_latent)
        
        
    def forward(self, x):
        x = self.lin1(x) 
        mu = self.lin_mu(x)
        std = nn.functional.softplus(self.lin_var(x))
        return mu, std

    
class Decoder(nn.Module):
    def __init__(self, input_size, n_hidden, n_latent):
        super().__init__()
        self.lin1 = nn.Linear(n_latent, n_hidden)
        self.lin2 = nn.Linear(n_hidden, input_size)
        self._input_size = input_size

        
    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        return x

    
class VAE(nn.Module):
    def __init__(self, input_size, n_hidden, n_latent):
        super().__init__()
        self.encoder = Encoder(input_size, n_hidden, n_latent)
        self.decoder = Decoder(input_size, n_hidden, n_latent)


    def forward(self, x):
        mu, std = self.encoder(x)
        z = torch.randn_like(std)*std + mu
        x = self.decoder(z)
        return x, mu, std


class Evaluator_VAE(Evaluator):
    def __init__(self, model, opt, beta=1e-4):
        super(Evaluator_VAE, self).__init__(
            model, opt, ("loss", "l_out", "l_kld")
        )
        self.beta = beta


    def loss_func(self, x, y, backward=True):
        y_pred, mu, std = self.model(x)
        l_out = self.out_loss(y, y_pred)
        l_kld = self.kld_loss(y, y_pred, mu, std)
        loss = l_out + self.beta*l_kld
        return loss, l_out, l_kld


    def out_loss(self, y_true, y_pred):
        delta = y_true - y_pred
        return torch.mean(delta*delta)


    def kld_loss(self, y_true, y_pred, mu, std):
        return .5*torch.mean(std*std + mu*mu - 1. - 2.*torch.log(std))

