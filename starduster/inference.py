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
    def __init__(self, sed_model, log_like, log_prior=None, output_mode='none', negative=False):
        super().__init__()
        self.sed_model = sed_model
        self.log_like = log_like
        if log_prior is None:
            self.log_prior = lambda *args: 0.
        else:
            self.log_prior = log_prior
        self.set_output_mode(output_mode, negative)


    def forward(self, x_in):
        if self._output_mode == 'numpy_grad':
            x_in = torch.tensor(x_in, dtype=torch.float32, requires_grad=True)
        y = self.sed_model(x_in)
        #free_params = self.sed_model.adapter.preprocess(x_in)
        log_post = self._sign*self.log_like(y)# + self.log_prior(*free_params)
        if self._output_mode == 'numpy':
            return log_post.detach().numpy()
        elif self._output_mode == 'numpy_grad':
            log_post.backward()
            return log_post.detach().numpy(), np.array(x_in.grad, dtype=np.float64)
        return log_post
    

    def set_output_mode(self, output_mode='torch', negative=False):
        if output_mode in ['torch', 'numpy', 'numpy_grad']:
            self._output_mode = output_mode
        else:
            raise ValueError(f"Unknown output mode: {output_mode}.")
        if negative:
            self._sign = -1.
        else:
            self._sign = 1.

