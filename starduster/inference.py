import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


class ErrorFunction(nn.Module):
    def __init__(self, param_names=None, bounds=None):
        if param_names is None:
            self.param_names = []
            self.bounds = np.empty((0, 2), dtype=np.float64)
        else:
            self.param_names = param_names
            self.bounds = bounds
        super().__init__()


class Gaussian(ErrorFunction):
    def __init__(self, y_obs, y_err, norm=True):
        super().__init__()
        self.register_buffer('y_obs', y_obs)
        self.register_buffer('y_err', y_err)
        if norm:
            self.norm = torch.sum(-torch.log(np.sqrt(2*np.pi)*y_err))
        else:
            self.norm = 0.


    def forward(self, *args):
        delta = (args[0] - self.y_obs)/self.y_err
        return torch.sum(-.5*delta*delta, dim=-1) + self.norm


class GaussianWithScatter(ErrorFunction):
    def __init__(self, y_obs, bounds=(-2., 0.)):
        super().__init__(['sigma'], bounds)
        self.register_buffer('y_obs', y_obs)
    

    def forward(self, *args):
        y_pred, log_sigma = args
        sigma = 10**log_sigma
        delta = (y_pred - self.y_obs)/sigma
        return torch.sum(-.5*delta*delta, dim=-1) \
            - self.y_obs.size(0)*torch.log(np.sqrt(2*np.pi)*sigma)


class Posterior(nn.Module):
    def __init__(self, sed_model, error_func, log_prior=None):
        super().__init__()
        self.sed_model = sed_model
        self.error_func = error_func
        if log_prior is None:
            self.log_prior = lambda *args: 0.
        else:
            self.log_prior = log_prior
        self.configure_output_mode()


    def forward(self, params):
        if self._output_mode == 'numpy_grad':
            params = torch.tensor(params, dtype=torch.float32, requires_grad=True)
        
        model_input_size = self.sed_model.input_size
        p_model = params[:model_input_size]
        p_error = params[model_input_size:]
        y_pred, is_out = self.sed_model(p_model, return_ph=True, check_bounds=True)
        log_post = self._sign*(self.error_func(y_pred, p_error) + self.log_out*is_out)

        if self._output_mode == 'numpy':
            return np.squeeze(log_post.detach().cpu().numpy())
        elif self._output_mode == 'numpy_grad':
            log_post.backward()
            return log_post.detach().cpu().numpy(), np.array(params.grad.cpu(), dtype=np.float64)
        return log_post


    def configure_output_mode(self, output_mode='torch', negative=False, log_out=-1e15):
        if output_mode in ['torch', 'numpy', 'numpy_grad']:
            self._output_mode = output_mode
        else:
            raise ValueError(f"Unknown output mode: {output_mode}.")
        if negative:
            self._sign = -1.
        else:
            self._sign = 1.
        self.log_out = log_out


    @property
    def input_size(self):
        """Number of input parameters."""
        return self.sed_model.adapter.input_size + len(self.error_func.param_names)


    @property
    def param_names(self):
        """Parameter names"""
        return self.sed_model.adapter.param_names + self.error_func.param_names


    @property
    def bounds(self):
        """Bounds of input parameters."""
        return np.vstack([self.sed_model.adapter.bounds, self.error_func.bounds])


class OptimizerWrapper(nn.Module):
    def __init__(self, log_post, x0):
        super().__init__()
        self.params = nn.Parameter(x0)
        # Save log_post as a tuple to prevent addtional parameters
        self._log_post = log_post,


    def forward(self):
        return self._log_post[0](self.params)


def optimize(log_post, cls_opt, x0=None, n_step=1000, lr=1e-2, progress_bar=True, **kwargs_opt):
    model = OptimizerWrapper(log_post, x0)
    opt = cls_opt(model.parameters(), lr=lr, **kwargs_opt)
    with tqdm(total=n_step, disable=(not progress_bar)) as pbar:
        for i_step in range(n_step):
            loss = model()
            loss.backward()
            opt.step()
            opt.zero_grad()

            pbar.set_description('loss: %.3e'%float(loss))
            pbar.update()

    return model.params.detach()

