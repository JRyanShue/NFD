

import numpy as np
import torch as th
from neural_field_diffusion.guided_diffusion import dist_util, logger

def unnormalize(sample, stats_dir):
    # raise Exception('This version of unnormalize is deprecated.')
    # RESCALE IMAGE -- Needs to be aligned with input normalization!
    min_values = np.load(f'{stats_dir}/lower_bound.npy').astype(np.float32).reshape(1, -1, 1, 1)  # should be (1, 96, 1, 1)
    max_values = np.load(f'{stats_dir}/upper_bound.npy').astype(np.float32).reshape(1, -1, 1, 1)
    _range = th.Tensor((max_values - min_values)).to(dist_util.dev())
    middle = th.Tensor(((min_values + max_values) / 2)).to(dist_util.dev())
    # print(sample.shape)  # ex: [4, 96, 128, 128]
    sample = sample * (_range / 2) + middle
    return sample