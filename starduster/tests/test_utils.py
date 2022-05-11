import torch
from numpy.testing import assert_raises
import starduster


def test_eval_batch():
    test_func = starduster.eval_batch(lambda a, b, c: a + b, batch_size=2, device='cpu')

    n_samp = 5
    n_dim = 2
    a = torch.rand(n_samp, n_dim)
    b = torch.rand(n_samp, n_dim)
    
    assert test_func(a, b, c=None).shape == a.shape
    assert test_func(a, b=b, c=None).shape == a.shape
    assert test_func(a=a, b=b, c=None).shape == a.shape
    assert_raises(ValueError, test_func, 2, 5, None)

