import torch
import numpy as np
from scipy.stats import truncnorm

class PositiveNormal:
    def __init__(self, locs, stds, dtype=torch.float32):
        assert locs.shape == stds.shape
        assert len(locs.shape) == len(stds.shape) == 1

        self.rvs = [truncnorm(- locs[i] / stds[i], np.inf, loc=locs[i], scale=stds[i]) for i in range(locs.shape[0])]
        self.dtype = dtype
    
    def sample(self, shape):
        assert len(shape) == 1
        return torch.stack([torch.tensor(d.rvs(shape[0]), dtype=self.dtype) for d in self.rvs], dim=1)