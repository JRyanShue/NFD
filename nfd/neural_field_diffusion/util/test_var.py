

import numpy as np
from tqdm import tqdm

from guided_diffusion.image_datasets import (
    ImageDataset, 
    _list_image_files_recursively
)

data_dir = '/home/jrshue/eg3d/eg3d/eg3d_triplane_ds'

ds = ImageDataset(128, _list_image_files_recursively(data_dir), normalize=False)

var_arr = np.full((15001, 128, 128), 0)
for idx, triplane in tqdm(enumerate(ds)):
    triplane = triplane[0]
    var_arr[idx] = triplane[0]  # Add first triplane

print(var_arr.shape)
print(f'VAR: {np.var(var_arr)}; SD: {np.std(var_arr)}')