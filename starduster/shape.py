from .utils import Evaluator

import torch


__all__ = ["Evaluator_Shape"]


class Evaluator_Shape(Evaluator):
    def __init__(self, model, opt):
        super(Evaluator_Shape, self).__init__(model, opt, ("loss",))


    def loss_func(self, x, y, backward=True):
        y_mu = y[:, :, 0]
        y_std = y[:, :, 1]
        y_pred = self.model(x)
        delta = (y_pred - y_mu)/y_std
        return torch.mean(delta*delta)

