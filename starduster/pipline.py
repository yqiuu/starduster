import torch


__all__ = ["Pipline"]


class Pipline:
    def __init__(self, keys, props):
        assert keys[0] == 'theta'

        self.keys = keys
        self.log_min = torch.min(torch.log10(props[:, 1:]), dim=0)[0]
        self.log_max = torch.max(torch.log10(props[:, 1:]), dim=0)[0]
        self.log_diff = self.log_max - self.log_min


    def transform(self, props):
        x = torch.zeros_like(props)
        x[:, 0] = torch.cos(props[:, 0])
        x[:, 1:] = (torch.log10(props[:, 1:]) - self.log_min)/self.log_diff
        x = 2*x - 1
        return x


    def inverse_tfansform(self, x):
        x = .5*(x + 1)
        props = torch.zeros_like(x)
        props[:, 0] = torch.arccos(x[:, 0])
        props[:, 1:] = 10**(self.log_diff*x[:, 1:] + self.log_min)
        return props
       
