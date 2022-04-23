import numpy as np
from scipy.optimize import minimize
import torch
from torch import nn
import starduster


class TestModule(nn.Module):
    def forward(self, x):
        return torch.sum(x*x, dim=-1)


def test_adpat_external():
    x0 = np.array([-2.3, 11.6])

    target_func = starduster.adapt_external(TestModule(), 'numpy')
    res = minimize(target_func, x0=x0, method='Nelder-Mead')
    assert isinstance(target_func(x0), np.ndarray)
    assert res.success
    assert np.allclose(res.x, 0., atol=1e-4)
    x_test = np.ones([5, 3])
    assert len(target_func(x_test)) == len(x_test)

    target_func = starduster.adapt_external(TestModule(), 'numpy_grad')
    res = minimize(target_func, x0=x0, method='L-BFGS-B', jac=True)
    assert isinstance(target_func(x0)[0], np.ndarray)
    assert isinstance(target_func(x0)[1], np.ndarray)
    assert res.success
    assert np.allclose(res.x, 0., atol=1e-4)

