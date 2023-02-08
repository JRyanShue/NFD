

import torch


def edr_loss(obj_idx, auto_decoder, device='cuda', offset_distance=0.01):

    num_points = 10000
    random_coords = torch.rand(obj_idx.shape[0], num_points, 3).to(device) * 2 - 1 # sample from [-1, 1]
    offset_coords = random_coords + torch.randn_like(random_coords) * offset_distance # Make offset_magnitude bigger if you want smoother
    densities_initial = auto_decoder(obj_idx, random_coords)
    densities_offset = auto_decoder(obj_idx, offset_coords)
    density_smoothness_loss = torch.nn.functional.mse_loss(densities_initial, densities_offset)

    return density_smoothness_loss
