import torch
from torch import nn
from torch.nn import functional as F
from numpy import pi


__all__ = [
    "Monotonic", "Unimodal", "Smooth", "PlankianMixture", "Forest", "LInfLoss",
    "CrossEntropy", "create_mlp", "kld_trapz", "kld_binary", "reduce_loss"
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
    def __init__(self, input_size, output_size):
        super().__init__()
        self.lin1 = nn.Linear(input_size, output_size)
        self.lin2 = nn.Linear(input_size, output_size)
        self.increase = Monotonic(increase=True)
        self.decrease = Monotonic(increase=False)


    def forward(self, x_in):
        return self.increase(self.lin1(x_in))*self.decrease(self.lin2(x_in))


class Smooth(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size


    def forward(self, x_in):
        x = F.avg_pool1d(x_in[:, None, :], self.kernel_size, 1)
        return x[:, 0, :]


class PlankianMixture(nn.Module):
    def __init__(self, input_size, n_mix, x):
        super().__init__()
        self.lin_mu = nn.Linear(input_size, n_mix)
        self.lin_w = nn.Linear(input_size, n_mix)
        self.const = 15/pi**4
        self.x_inv = 1./x


    def plank(self, mu):
        y = self.x_inv*mu[:, :, None]
        f = torch.exp(-y)
        return self.const*y**4*f/(1 - f)


    def forward(self, x_in):
        mu = torch.cumsum(torch.exp(self.lin_mu(x_in)), dim=1)
        w = F.softmax(self.lin_w(x_in), dim=1)
        return torch.sum(self.plank(mu)*w[:, :, None], dim=1)


class Forest(nn.Module):
    def __init__(self, input_size, forest_size, dx):
        super().__init__()
        self.lin_pos = nn.Linear(input_size, forest_size)
        self.lin_neg = nn.Linear(input_size, forest_size)
        self.dx = dx


    def forward(self, x_in, continuum):
        f_pos = torch.exp(self.lin_pos(x_in))
        f_pos = f_pos/torch.trapz(f_pos, dx=self.dx)[:, None]
        f_neg = -torch.sigmoid(self.lin_neg(x_in))*continuum
        return -torch.trapz(f_neg, dx=self.dx)[:, None]*f_pos + f_neg


class LInfLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction


    def forward(self, y_true, y_pred):
        loss = torch.linalg.norm(y_pred - y_true, ord=float('inf'), dim=1)
        return reduce_loss(loss, self.reduction)


class CrossEntropy(nn.Module):
    def __init__(self, dx, eps=1e-20, reduction='mean'):
        super().__init__()
        self.dx = dx
        self.eps = eps
        self.reduction = reduction


    def __call__(self, y_pred, y_true):
        loss = -torch.trapz(y_true*torch.log((y_pred + self.eps)/y_true), dx=self.dx)
        return reduce_loss(loss, self.reduction)


def create_mlp(input_size, layer_sizes, activations):
    modules = []
    size_in = input_size
    for size_out, act in zip(layer_sizes, activations):
        modules.append(nn.Linear(size_in, size_out))
        if act is not None:
            modules.append(act)
        size_in = size_out
    return nn.Sequential(*modules)


def kld_trapz(a_pred, a_true, dx, eps=1e-10):
    """Compute KL divergence using the trapezoidal rule."""
    return -torch.trapz(a_true*torch.log((a_pred + eps)/a_true), dx=dx)


def kld_binary(a_pred, a_true, eps=1e-10):
    """Compute binary KL divergence."""
    a_pred = F.hardtanh(a_pred, eps, 1 - eps)
    b_pred = 1 - a_pred
    b_true = 1 - a_true
    return -a_true*torch.log(a_pred/a_true) - b_true*torch.log(b_pred/b_true)


def reduce_loss(loss, reduction):
    if reduction is 'mean':
        return torch.mean(loss)
    elif reduction is 'sum':
        return torch.sum(loss)
    elif reduction is 'square_mean':
        return torch.mean(loss*loss)
    elif reduction is 'square_sum':
        return torch.sum(loss*loss)
    elif reduction is 'none':
        return loss
    else:
        raise ValueError("Invalid reduction: {}".format(reduction))

