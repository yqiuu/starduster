import numpy as np
import torch


class Helper:
    def __init__(self, header, lookup):
        self.header = header
        self.lookup = lookup


    def get_item(self, target, key):
        return target[:, self.lookup[key]]


    def set_item(self, target, key, value):
        target[:, self.lookup[key]] = value
        return target


    def transform(self, target, key=None):
        if key is None:
            key = list(self.header.keys())
        else:
            key = np.ravel(key)
        output = torch.zeros([target.size(0), len(key)], dtype=target.dtype, device=target.device)
        for i_k, k in enumerate(key):
            arr = self.get_item(target, k)
            if k == 'theta':
                arr = torch.cos(np.pi/180.*arr)
            elif k == 'b_to_t':
                pass
            else:
                log_min, log_max = self.header[k]
                arr = (torch.log10(arr) - log_min)/(log_max - log_min)
            output[:, i_k] = arr
        output = torch.squeeze(output)
        output = 2*output - 1 # Convert from [0, 1] to [-1, 1]
        return output


    def recover(self, target, key=None):
        if key is None:
            key = list(self.header.keys())
        else:
            key = np.ravel(key)
        output = torch.zeros([target.size(0), len(key)], dtype=target.dtype, device=target.device)
        for i_k, k in enumerate(key):
            arr = self.get_item(target, k)
            arr = .5*(arr + 1) # Convert from [-1, 1] to [0, 1]
            if k == 'theta':
                arr = 180/np.pi*np.arccos(arr)
            elif k == 'b_to_t':
                pass
            else:
                log_min, log_max = self.header[k]
                arr = 10**(arr*(log_max - log_min) + log_min)
            output[:, i_k] = arr
        output = torch.squeeze(output)
        return output

