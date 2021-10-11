from .selector import sample_from_selector

import torch


class Analyzer:
    def __init__(self, sed_model):
        self.sed_model = sed_model


    def sample(self, n_samp):
        adpater = self.sed_model.adapter
        lb, ub = torch.tensor(self.sed_model.adapter.bounds, dtype=torch.float32).T
        sampler = lambda n_samp: (ub - lb)*torch.rand([n_samp, len(lb)]) + lb
        return sample_from_selector(n_samp, adpater.selector_disk, adpater.selector_bulge, sampler)

