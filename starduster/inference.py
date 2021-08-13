import torch
from torch import nn
from torch.nn import functional as F


class GaussianLikelihood(nn.Module):
    def __init__(self, y_obs, y_err):
        super().__init__()
        self.multi_normal = torch.distributions.MultivariateNormal(y_obs, torch.diag(y_err*y_err))

        
    def forward(self, y):
        return self.multi_normal.log_prob(y)
    
    
class Preprocess(nn.Module):
    def forward(self, params, sfh_disk):
        params = F.hardtanh(params)
        sfh_disk = F.softmax(sfh_disk, dim=-1)
        return params, sfh_disk
        

class Posterior(nn.Module):
    def __init__(self, sed_model, log_like, log_prior=None, preprocess=None):
        super().__init__()
        self.sed_model = sed_model
        self.log_like = log_like
        if log_prior is None:
            self.log_prior = lambda *args: 0.
        else:
            self.log_prior = log_prior
        if preprocess is None:
            self.preprocess = lambda x: x
        else:
            self.preprocess = preprocess
    

    def forward(self, x_in):
        x = self.preprocess(*self.unflatten(torch.atleast_2d(x_in)))
        y = self.sed_model(*x)
        return self.log_like(y) + self.log_prior(*x)
    
    
    def unflatten(self, x_in):
        input_shapes = self.sed_model.input_shapes
        x_out = [None]*len(input_shapes)
        idx_b = 0
        for i_input, size in enumerate(input_shapes):
            idx_e = idx_b + size
            x_out[i_input] = x_in[:, idx_b:idx_e]
            idx_b = idx_e
        return x_out

