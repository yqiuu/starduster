import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


class GaussianLikelihood(nn.Module):
    def __init__(self, y_obs, y_err, norm=True):
        super().__init__()
        self.register_buffer('y_obs', y_obs)
        self.register_buffer('y_err', y_err)
        if norm:
            self.norm = torch.sum(-torch.log(np.sqrt(2*np.pi)*y_err))
        else:
            self.norm = 0.


    def forward(self, y):
        delta = (y - self.y_obs)/self.y_err
        return torch.sum(-.5*delta*delta, dim=-1) + self.norm


class Posterior(nn.Module):
    def __init__(self, sed_model, log_like, log_prior=None):
        super().__init__()
        self.sed_model = sed_model
        self.log_like = log_like
        if log_prior is None:
            self.log_prior = lambda *args: 0.
        else:
            self.log_prior = log_prior
        self.configure_output_mode()


    def forward(self, params):
        if self._output_mode == 'numpy_grad':
            params = torch.tensor(params, dtype=torch.float32, requires_grad=True)

        sed_model = self.sed_model
        gp, sfh_disk, sfh_bulge, is_out = sed_model.adapter.derive_free_params(params)
        model_params = sed_model.adapter.derive_model_params(gp, sfh_disk, sfh_bulge)

        gp = model_params[0]
        gp_curve_disk = sed_model.helper.get_item(gp, 'curve_disk_inds')
        gp_curve_bulge = sed_model.helper.get_item(gp, 'curve_bulge_inds')
        is_out_disk = ~sed_model.selector_disk.select(gp_curve_disk)
        is_out_bulge = ~sed_model.selector_bulge.select(gp_curve_bulge)
        log_out = self.log_out*(is_out | is_out_disk | is_out_bulge)

        y = sed_model.generate(*model_params)
        log_post = self._sign*(self.log_like(y) + log_out)
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


class OptimizerWrapper(nn.Module):
    def __init__(self, log_post, x0=None):
        super().__init__()
        if x0 is None:
            adapter = log_post.sed_model.adapter
            bound_centre = adapter.bound_centre
            bound_radius = adapter.bound_radius
            self.params = nn.Parameter(
                bound_centre + .5*bound_radius*(2*torch.rand_like(bound_centre) - 1)
            )
        else:
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

