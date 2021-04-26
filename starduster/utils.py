from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn


__all__ = ["Evaluator", "fit", "merge_history"]


class Evaluator(ABC):
    def __init__(self, model, opt, labels=None):
        self.model = model
        self.opt = opt
        if labels is None:
            self.labels = ("loss",)
        else:
            self.labels = labels


    @abstractmethod
    def loss_func(self, x, y):
        return 0.


    def add_scheduler(self, scheduler):
        self.scheduler = scheduler


    def call(self, data, backward=True):
        values = [None]*len(data)
        for i_b, d_b in enumerate(data):
            if backward:
                l_b = self.loss_func(*d_b)
                if isinstance(l_b, torch.Tensor):
                    l_b = [l_b]
                l_b[0].backward()
                self.opt.step()
                self.opt.zero_grad()
                try:
                    self.scheduler.step()
                except AttributeError:
                    pass
            else:
                self.model.eval()
                l_b = self.loss_func(*d_b)
                if isinstance(l_b, torch.Tensor):
                    l_b = [l_b]
            values[i_b] = l_b
        values = torch.mean(torch.tensor(values), dim=0).detach().tolist()
        return values


def fit(evaluator, dl_train, dl_valid, n_epochs=100):
    history = {'epoch':[0]*n_epochs}
    for name in evaluator.labels:
        history[name] = [0]*n_epochs
        history[f"{name}_val"] = [0]*n_epochs

    history_train = [None]*n_epochs
    history_valid = [None]*n_epochs
    for i_e in range(n_epochs):
        history_train[i_e] = evaluator.call(dl_train, backward=True)
        history_valid[i_e] = evaluator.call(dl_valid, backward=False)

        msg = ["{}(val)={:.2e}({:.2e})".format(*values) for values in \
            zip(evaluator.labels, history_train[i_e], history_valid[i_e])]
        msg = "\repoch={}, ".format(i_e + 1) + ", ".join(msg)
        print(msg, end="")
    print()

    history = {}
    history['epoch'] = np.arange(1, n_epochs + 1)
    for name, values in zip(evaluator.labels, zip(*history_train)):
        history[name] = np.asarray(values)
    for name, values in zip(evaluator.labels, zip(*history_valid)):
        history[f"{name}_val"] = np.asarray(values)

    return history


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

