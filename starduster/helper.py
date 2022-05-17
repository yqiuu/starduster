import numpy as np
import torch


class Helper:
    # Convert the parameter to [-1, 1]
    transform_func = {
        'theta': lambda x, a, b, p: 2*p.cos(np.pi/180.*x) - 1,
        'frac': lambda x, a, b, p: 2*x - 1,
        'log': lambda x, a, b, p: 2*(p.log10(x) - a)/(b - a) - 1
    }
    # Convert the parameter from [-1, 1] to the original value
    recover_func = {
        'theta': lambda x, a, b, p: 180/np.pi*p.arccos(.5*(x + 1)),
        'frac': lambda x, a, b, p: .5*(x + 1),
        'log': lambda x, a, b, p: 10**(.5*(x + 1)*(b - a) + a)
    }


    def __init__(self, header, lookup):
        self.header = header
        self.lookup = lookup


    def get_item(self, target, key):
        return target[:, self.lookup[key]]


    def set_item(self, target, key, value):
        target[:, self.lookup[key]] = value
        return target


    def transform(self, val, key, lib=np):
        key_trans, lb, ub = self.header[key]
        return self.transform_func[key_trans](val, lb, ub, lib)


    def recover(self, val, key, lib=np):
        key_trans, lb, ub = self.header[key]
        return self.recover_func[key_trans](val, lb, ub, lib)


    def get_transform(self, target, key, lib=np):
        return self.transform(self.get_item(target, key), key, lib)


    def get_recover(self, target, key, lib=np):
        return self.recover(self.get_item(target, key), key, lib)


    def transform_all(self, target, lib=np):
        output = lib.zeros_like(target)
        for i_k, (key, lb, ub) in enumerate(self.header.values()):
            output[:, i_k] = self.transform_func[key](target[:, i_k], lb, ub, lib)
        return output


    def recover_all(self, target, lib=np):
        output = lib.zeros_like(target)
        for i_k, (key, lb, ub) in enumerate(self.header.values()):
            output[:, i_k] = self.recover_func[key](target[:, i_k], lb, ub, lib)
        return output


    def transform_scale(self, key, scale):
        lb, ub = -1., 1.
        lb_r = self.recover(lb, key)
        ub_r = self.recover(ub, key)
        if self.header[key][0] == 'log':
            lb_r = np.log10(lb_r)
            ub_r = np.log10(ub_r)
        return abs((lb - ub)/(lb_r - ub_r)*scale)


    def recover_scale(self, key, scale):
        return scale/self.transform_scale(key, 1.)

