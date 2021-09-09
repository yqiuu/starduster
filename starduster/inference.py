import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


class GaussianLikelihood(nn.Module):
    def __init__(self, y_obs, y_err):
        super().__init__()
        self.multi_normal = torch.distributions.MultivariateNormal(y_obs, torch.diag(y_err*y_err))

        
    def forward(self, y):
        return self.multi_normal.log_prob(y)
    
    
class Posterior(nn.Module):
    def __init__(self,
        sed_model, log_like, log_prior=None, output_mode='none', negative=False, log_out=-1e15
    ):
        super().__init__()
        self.sed_model = sed_model
        self.log_like = log_like
        if log_prior is None:
            self.log_prior = lambda *args: 0.
        else:
            self.log_prior = log_prior
        self.set_output_mode(output_mode, negative)
        self.log_out = log_out


    def forward(self, params):
        if self._output_mode == 'numpy_grad':
            params = torch.tensor(params, dtype=torch.float32, requires_grad=True)

        sed_model = self.sed_model
        free_params, is_out = sed_model.adapter.derive_free_params(params, check_bounds=True)
        model_params = sed_model.adapter.derive_model_params(free_params)
        log_out = is_out*self.log_out
        y = sed_model.generate(*model_params)
        #free_params = self.sed_model.adapter.preprocess(params)
        log_post = self._sign*(self.log_like(y) + log_out)
        if self._output_mode == 'numpy':
            return log_post.detach().cpu().numpy()
        elif self._output_mode == 'numpy_grad':
            log_post.backward()
            return log_post.detach().cpu().numpy(), np.array(params.grad.cpu(), dtype=np.float64)
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


class OptimizerWrapper(nn.Module):
    def __init__(self, log_post, x0=None):
        super().__init__()
        if x0 is None:
            self.params = nn.Parameter(torch.full((1, log_post.sed_model.adapter.input_size), .5))
        else:
            self.params = nn.Parameter(x0)
        # Save log_post as a tuple to prevent addtional parameters
        self._log_post = log_post,


    def forward(self):
        return self._log_post[0](self.params)


def optimize(log_post, cls_opt, x0=None, n_step=1000, lr=1e-2, **kwargs_opt):
    model = OptimizerWrapper(log_post, x0)
    opt = cls_opt(model.parameters(), lr=lr, **kwargs_opt)
    with tqdm(total=n_step) as pbar:
        for i_step in range(n_step):
            loss = model()
            loss.backward()
            opt.step()
            opt.zero_grad()

            pbar.set_description('loss: %.3e'%float(loss))
            pbar.update()

    return model.params.detach()

