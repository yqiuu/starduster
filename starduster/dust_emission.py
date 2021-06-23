from .utils import create_mlp

import torch
from torch import nn


class AttenuationFraction(nn.Module):
    def __init__(self, input_size, layer_sizes, activations):
        super().__init__()
        self.mlp = create_mlp(input_size, layer_sizes, activations)
    
        
    def forward(self, x):
        return torch.sum(self.mlp(x[0])*x[1], dim=1)
