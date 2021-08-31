import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class GaussianLikelihood(nn.Module):
    def __init__(self, y_obs, y_err):
        super().__init__()
        self.multi_normal = torch.distributions.MultivariateNormal(y_obs, torch.diag(y_err*y_err))

        
    def forward(self, y):
        return self.multi_normal.log_prob(y)
    
    
class Posterior(nn.Module):
    def __init__(self, sed_model, log_like, log_prior=None, output_mode='none'):
        super().__init__()
        self.sed_model = sed_model
        self.log_like = log_like
        if log_prior is None:
            self.log_prior = lambda *args: 0.
        else:
            self.log_prior = log_prior
        self.output_mode = output_mode
    

    def forward(self, x_in):
        if self.output_mode == 'numpy_grad':
            x_in = torch.tensor(x_in, dtype=torch.float32, requires_grad=True)
        y = self.sed_model(x_in)
        free_params = self.sed_model.adapter.preprocess(x_in)
        log_post = self.log_like(y)# + self.log_prior(*free_params)
        if self.output_mode == 'numpy':
            return log_post.detach().numpy()
        elif self.output_mode == 'numpy_grad':
            log_post.backward()
            return log_post.detach().numpy(), np.array(x_in.grad, dtype=np.float64)
        return log_post
    
