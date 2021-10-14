from collections import namedtuple

import numpy as np
import torch


__all__ = [
    "constants", "units", "merge_history", "load_model", "search_inds",
    "reduction", "interp_arr", "accept_reject"
]

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
        'sfr_10': 'M_sol/yr',
        'sfr_100': 'M_sol/yr'
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
    checkpoint = torch.load(fname, map_location=map_location)
    model = init(*checkpoint['params'])
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint


def search_inds(x, a, b):
    "Find the indices of x that are closest to the given points."
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
            I =  simps(y[ia], y[im], y[ib], x[ia], x[im], x[ib])
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


def interp_arr(x, xp, yp, left=None, right=None, period=None):
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
    return samps_accept[:n_samp]

