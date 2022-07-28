from .utils import accept_reject, adapt_external
from .sed_model import MultiwavelengthSED

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


def create_posterior(
    sed_model, noise_model, prior_model=None, mode='numpy', negative=False, device=None
):
    """Create a posterior function that can be passed to various optimisation
    and sampling tools.

    Parameters
    ----------
    sed_model : MultiwavelengthSED
        The SED model.
    noise_model : module
        A noise model that includes the observational data.
    prior_model : module
        A prior distribution.
    mode : str {'torch', 'numpy', 'numpy_grad'}
        | 'torch': Create a function that accepts PyTorch tensors.
        | 'numpy': Create a function that accepts NumPy arrays.
        | 'numpy_grad': Create a function that accepts NumPy arrays and
        includes the gradient with respect to the input as the second return
        parameter.

    negative : bool
        Set ``negative=True`` if the sampler is a minimizer.
    device : str
        Device for the required modules.
    """
    if device is not None:
        sed_model.to(device)
        noise_model.to(device)
        prior_model.to(device)

    def target_func(params):
        x_pred, (log_p_in_disk, log_p_in_bulge) \
            = sed_model(params, return_ph=True, check_selector=True)
        x_pred = torch.atleast_2d(x_pred)
        log_prob = noise_model.log_prob(x_pred) + log_p_in_disk + log_p_in_bulge
        if prior_model is not None:
            log_prob = log_prob + prior_model.log_prob(params)
        return log_prob

    return adapt_external(target_func, mode=mode, negative=negative, device=device)


class IndependentNormal(nn.Module):
    def __init__(self, mu, sigma, is_normed=True):
        super().__init__()
        self.register_buffer('mu', mu)
        self.register_buffer('sigma', sigma)
        self.register_buffer('log_norm', -torch.sum(.5*torch.log(2*math.pi*sigma*sigma)))
        self.is_normed = is_normed


    def sample(self, size):
        mu = self.mu
        return mu + self.sigma*torch.randn(*size, mu.size(0), device=mu.device)


    def log_prob(self, x):
        delta = (x - self.mu)/self.sigma
        log_prob = -.5*torch.sum(delta*delta, dim=-1, keepdim=True)
        if self.is_normed:
            log_prob += self.log_norm
        return log_prob


class SmoothedUniform(nn.Module):
    def __init__(self, bounds, eps=1e-3):
        super().__init__()
        lb, ub = bounds.T
        self.register_buffer('lb', lb)
        self.register_buffer('ub', ub)
        self.register_buffer('alpha', math.pi/(eps*(ub - lb))**2)
        self.register_buffer('log_norm', -torch.sum(torch.log((1 + eps)*(ub - lb))))


    def log_prob(self, x):
        cond_1 = x < self.lb
        cond_2 = x >= self.ub
        d1 = (x - self.lb)
        d2 = (x - self.ub)
        return torch.sum(-self.alpha*(d1*d1*cond_1 + d2*d2*cond_2), dim=-1, keepdim=True) \
            + self.log_norm


class OptimizerWrapper(nn.Module):
    """A wrapper that allows a posterior distribution to be minimised by
    PyTorch optimizers.

    Parameters
    ----------
    log_post : Posterior
        Target posterior distribution.
    x0 : tensor
        Initial parameters.

    Output
    ------
        Scalar.
    """
    def __init__(self, log_post, x0):
        super().__init__()
        self.params = nn.Parameter(x0)
        # Save log_post as a tuple to prevent addtional parameters
        self._log_post = log_post,


    def forward(self):
        return self._log_post[0](self.params)


def optimize(log_post, cls_opt, x0=None, n_step=1000, lr=1e-2, progress_bar=True, **kwargs_opt):
    """Optimise a posterior distribution using a Pytorch optimiser.

    Parameters
    ----------
    log_post : Posterior
        Target posterior distribution.
    cls_opt : torch.optim.Optimizer
        PyTorch optimiser class.
    x0 : tensor
        Initial parameters.
    n_step : int
        Number of the optimisation steps.
    lr : float
        Learning rate.
    progress_bar : bool
        If True, show a progress bar.
    kwargs_opt
        Keyword arguments used to initalise the optimiser.

    Returns
    -------
    tensor
        Best-fitting parameters.
    """
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


def optimize(log_post, cls_opt, x0=None, n_step=1000, lr=1e-2, progress_bar=True, **kwargs_opt):
    """Optimise a posterior distribution using a Pytorch optimiser.

    Parameters
    ----------
    log_post : Posterior
        Target posterior distribution.
    cls_opt : torch.optim.Optimizer
        PyTorch optimiser class.
    x0 : tensor
        Initial parameters.
    n_step : int
        Number of the optimisation steps.
    lr : float
        Learning rate.
    progress_bar : bool
        If True, show a progress bar.
    **kwargs_opt
        Keyword arguments used to initalise the optimiser.

    Returns
    -------
    tensor
        Best-fitting parameters.
    """
    params = torch.tensor(x0, device=x0.device, requires_grad=True)
    opt = cls_opt([params], lr=lr, **kwargs_opt)
    with tqdm(total=n_step, disable=(not progress_bar)) as pbar:
        for i_step in range(n_step):
            loss = log_post(params)
            loss.backward()
            opt.step()
            opt.zero_grad()

            pbar.set_description('loss: %.3e'%float(loss))
            pbar.update()

    return params.detach()


def sample_effective_region(target, n_samp=1, sampler=None, max_iter=10000):
    """Sample some parameters that can be passed to the target.

    The resulting samples are within the bounds and can be accepted by the
    selectors.

    Parameters
    ----------
    target : MultiwavelengthSED or Posterior
        A MultiwavelengthSED or Posterior instance.
    n_samp : int
        Number of the samples
    sampler : callable
        A base sampler. If ``None``, sample parameters uniformly in the bounds.
    max_iter : int
        Maximum iteration of the accept-reject sampling.

    Returns
    -------
    samps : tensor
        Parameters that can be passed to the target.
    """
    if isinstance(target, MultiwavelengthSED):
        sed_model = target
        posterior = None
    else:
        posterior = target
        sed_model = target.sed_model

    adapter = sed_model.adapter
    if not adapter.flat_input:
        raise ValueError("Set flat_input to be true.")

    if posterior is None:
        n_col = sed_model.input_size
        bounds = sed_model.bounds
    else:
        n_col = posterior.input_size
        bounds = posterior.bounds
    lb, ub = torch.tensor(bounds, dtype=torch.float32).T

    if sampler is None:
        sampler = lambda n_samp: (ub - lb)*torch.rand([n_samp, n_col]) + lb

    helper = sed_model.helper
    def condition(params):
        cond = torch.all(params > lb, dim=-1) & torch.all(params < ub, dim=-1)
        gp = adapter(params)[0]
        cond &= adapter.selector_disk.select(helper.get_item(gp, 'curve_disk_inds')) \
            & adapter.selector_bulge.select(helper.get_item(gp, 'curve_bulge_inds'))
        return cond

    device = adapter.device
    try:
        adapter.cpu()
        samps = accept_reject(n_samp, n_col, sampler, condition, max_iter)
    finally:
        adapter.to(device)

    return samps

