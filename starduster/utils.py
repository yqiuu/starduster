from collections import namedtuple

import numpy as np
import torch
from torch import nn


__all__ = [
    "Configurable", "Regrid",
    "constants", "units", "merge_history", "load_model", "search_inds",
    "reduction", "interp_arr", "accept_reject", "adapt_external"
]


class Configurable:
    def __init__(self, **default_config):
        self._config_names = list(default_config.keys())
        self.configure(**default_config)


    def configure(self, **kwargs):
        self.set_config(**kwargs)
        self.update_config()


    def get_config(self):
        return {name: getattr(self, name) for name in self._config_names}


    def set_config(self, **kwargs):
        for key, val in kwargs.items():
            if key in self._config_names:
                setattr(self, key, val)
            else:
                raise ValueError(f"Unknow config: {key}.")


    def update_config(self):
        pass


class Regrid(nn.Module):
    """Apply linear interpolation to an array of y data assuming the same x
    data.
    """
    def __init__(self, x_eval, x_data, fill_value=0.):
        inds = torch.searchsorted(x_data, x_eval)
        inds[inds == 0] = 1 # Only happen when min(x_eval) = min(x_data)
        inds[inds == x_data.size(0)] = x_data.size(0) - 1
        x0 = x_data[inds - 1]
        x1 = x_data[inds]
        weights = (x_eval - x0)/(x1 - x0)
        mask = torch.as_tensor((x_eval >= x_data[0]) & (x_eval <= x_data[-1]), dtype=x_data.dtype)
        
        super().__init__()
        self.register_buffer('inds', inds)
        self.register_buffer('weights', weights)
        self.register_buffer('mask', mask)
        self.register_buffer('fill_value', torch.tensor(fill_value, dtype=x_data.dtype))
        
        
    def forward(self, y_data):
        y0 = y_data[..., self.inds - 1]
        y1 = y_data[..., self.inds]
        y_eval = torch.lerp(y0, y1, self.weights)
        y_eval = self.mask*y_eval + (1 - self.mask)*self.fill_value
        return y_eval


def namedtuple_from_dict(name, target):
    return namedtuple(name, target.keys())(**target)

constants = namedtuple_from_dict(
    'Constants',
    {'met_sol': 0.019}
)

units = namedtuple_from_dict(
    'Units',
    {
        'theta': 'deg',
        'r_disk': 'kpc',
        'r_bulge': 'kpc',
        'r_dust': 'kpc',
        'l_norm': 'L_sol',
        'b_to_t': '',
        'm_dust': 'M_sol',
        'm_disk': 'M_sol',
        'm_bulge': 'M_sol',
        'm_star': 'M_sol',
        'sfr': 'M_sol/yr',
        'met': 'Z_sol'
    }
)

def merge_history(history1, history2):
    history = {}
    for key in history1:
        if key == 'epoch':
            history[key] = np.append(
                history1[key], history1[key][-1] + history2[key]
            )
        else:
            history[key] = np.append(
                history1[key], history2[key]
            )
    return history


def load_model(fname, init, map_location=None):
    """Load a model.

    Parameters
    ----------
    fname : str
        File name of the model.
    init : object
        Initialiser of the model.
    map_location
        A variable that is passed to torch.load.
    """
    checkpoint = torch.load(fname, map_location=map_location)
    model = init(*checkpoint['params'])
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint


def search_inds(x, a, b):
    """Find the indices of x that are closest to the given points."""
    return np.argmin(np.abs(x - a)), np.argmin(np.abs(x - b)) + 1


def reduction(y, x, eps=5e-4):
    """Reduce the number of input data points using integral."""
    def split(ia, ib):
        split_list = []
        done_list = []

        I_fid = np.sum(blocks[ia:ib], axis=0)
        if ib - ia == 1:
            done_list.append([ia, ib, I_fid])
        elif ib - ia == 2:
            im = ia + 1
            I = simps(y[ia], y[im], y[ib], x[ia], x[im], x[ib])
            if np.all(np.abs(I_fid - I) < eps) :
                done_list.append([ia, ib, I])
            else:
                done_list.append([ia, im, blocks[ia]])
                done_list.append([im, ib, blocks[im]])
        elif ib - ia > 2:
            im = (ia + ib)//2
            I = simps(y[ia], y[im], y[ib], x[ia], x[im], x[ib])
            if np.all(np.abs(I_fid - I) < eps):
                done_list.append([ia, ib, I])
            else:
                split_list.append([ia, im])
                split_list.append([im, ib])
        else:
            raise ValueError

        return split_list, done_list


    y = np.atleast_2d(y).T
    blocks = .5*(y[:-1] + y[1:])*np.diff(x)[:, None]
    split_list = [[0, len(blocks)]]

    inds_out = np.full(len(x), -1, dtype='i4')
    x_out = np.full(len(x) - 1, np.nan)
    y_out = np.full_like(blocks, np.nan)

    while len(split_list) > 0:
        split_list_next = []
        for ia, ib in split_list:
            split_list_sub, done_list = split(ia, ib)
            split_list_next.extend(split_list_sub)
            for i0, i1, I in done_list:
                inds_out[i0] = i0
                inds_out[i1] = i1
                y_out[i0] = I
                x_out[i0] = .5*(x[i0] + x[i1])
            split_list = split_list_next

    inds_out = inds_out[inds_out >= 0]
    cond = ~np.isnan(x_out)
    y_out = np.squeeze(y_out[cond].T)
    x_out = x_out[cond]
    return y_out, x_out, inds_out


def simps(y0, y1, y2, x0, x1, x2):
    h = x2 - x0
    h0 = x1 - x0
    h1 = x2 - x1
    return h/6.*((2. - h0/h1)*y0 + h*h/(h0*h1)*y1 + (2. - h1/h0)*y2)


def simple_trapz(y, x):
    """Apply the trapzoid rule without the final summation"""
    return .5*(y[:, 1:] + y[:, :-1])*(x[:, 1:] - x[:, :-1])


def interp_arr(x, xp, yp, left=None, right=None, period=None):
    """Apply linear interpolation to an array of y data assuming the same x
    data.
    """
    y_out = np.zeros([len(yp), len(x)])
    for i_y, y in enumerate(yp):
        y_out[i_y] = np.interp(x, xp, yp[i_y], left, right, period)
    return y_out


def accept_reject(n_samp, n_col, sampler, condition, max_iter=10000):
    samps_accept = torch.zeros([0, n_col])
    for it in range(max_iter):
        samps = sampler(n_samp)
        samps = samps[condition(samps)]
        samps_accept = torch.vstack([samps_accept, samps])
        if len(samps_accept) >= n_samp:
            break
    if len(samps_accept) < n_samp:
        raise ValueError("Maximum iteration is reached.")
    if n_samp == 1:
        return samps_accept[0]
    else:
        return samps_accept[:n_samp]


def adapt_external(func, mode='numpy', device='cpu'):
    def wrapper(params):
        if mode == 'numpy':
            params = torch.as_tensor(params, dtype=torch.float32, device=device)
            with torch.no_grad():
                val = np.squeeze(func(params).cpu().numpy())
            return val
        elif mode == 'numpy_grad':
            params = torch.tensor(params, dtype=torch.float32, device=device, requires_grad=True)
            val = func(params)
            val.backward()
            val = np.squeeze(val.cpu().detach().numpy())
            # There may be a issue when the gradient is not float64 for scipy algorithms
            val_grad = np.array(params.grad.cpu(), dtype=np.float64)
            return val, val_grad
        else:
            raise ValueError(f"Unknown mode: '{mode}'.")

    return wrapper

