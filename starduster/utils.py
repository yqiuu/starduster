import numpy as np
import torch


__all__ = ["merge_history", "load_module"]


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


def load_module(fname, cls):
    checkpoint = torch.load(fname)
    module = cls(*checkpoint['params'])
    module.load_state_dict(checkpoint['model_state_dict'])
    return module, checkpoint

