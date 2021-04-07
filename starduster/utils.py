from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn


class Evaluator(ABC):
    def __init__(self, model, opt, labels):
        self.model = model
        self.opt = opt
        self.labels = labels


    @abstractmethod
    def loss_func(self, x, y):
        return 0., None

    
    def call(self, data, backward=True):
        values = [None]*len(data)
        for i_b, (x_b, y_b) in enumerate(data):
            if backward:
                l_b = self.loss_func(x_b, y_b)
                l_b[0].backward()
                self.opt.step()
                self.opt.zero_grad()
            else:
                self.model.eval()
                l_b = self.loss_func(x_b, y_b)
            if l_b[1] is None:
                values[i_b] = [l_b[0]]
            else:
                values[i_b] = l_b
        values = torch.mean(torch.tensor(values), dim=0).detach().tolist()
        return values


def create_MLP(layers, acts):
    model = nn.Sequential()
    for i in range(len(acts)):
        model.add_module(f'lin{i}', nn.Linear(layers[i], layers[i+1]))
        if acts[i] == 'tanh':
            model.add_module(f'act{i}', nn.Tanh())
        elif acts[i] == 'softplus':
            model.add_module(f'act{i}', nn.Softplus())
    return model


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

