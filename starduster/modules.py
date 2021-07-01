import torch
from torch import nn
from torch.nn import functional as F


__all__ = [
    "LInfLoss", "Monotonic", "Unimodal", "Smooth", "create_mlp", "reduce_loss"
]


class Monotonic(nn.Module):
    def __init__(self, increase=True):
        super().__init__()
        self.increase = increase


    def forward(self, x_in):
        x = F.softplus(x_in)
        x = torch.cumsum(x, dim=1)/x.size(1)
        if self.increase:
            return x - x[:, None, 0]
        else:
            return -x + x[:, None, -1]


class Unimodal(nn.Module):
    def __init__(self):
        super().__init__()
        self.increase = Monotonic(increase=True)
        self.decrease = Monotonic(increase=False)


    def forward(self, x_in):
        return self.increase(x_in)*self.decrease(x_in)


class Smooth(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size


    def forward(self, x_in):
        x = F.avg_pool1d(x_in[:, None, :], self.kernel_size, 1)
        return x[:, 0, :]


class LInfLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction


    def forward(self, y_true, y_pred):
        loss = torch.linalg.norm(y_pred - y_true, ord=float('inf'), dim=1)
        return reduce_loss(loss, self.reduction)


def reduce_loss(loss, reduction):
    if reduction is 'mean':
        return torch.mean(loss)
    elif reduction is 'sum':
        return torch.sum(loss)
    elif reduction is 'none':
        return loss
    else:
        raise ValueError("Invalid reduction: {}".format(reduction))


def create_mlp(input_size, layer_sizes, activations):
    modules = []
    size_in = input_size
    for size_out, act in zip(layer_sizes, activations):
        modules.append(nn.Linear(size_in, size_out))
        if act is not None:
            modules.append(act)
        size_in = size_out
    return nn.Sequential(*modules)

