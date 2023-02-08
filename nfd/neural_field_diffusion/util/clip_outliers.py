
'''
From a ds of outliers, create new ds clipped to specified standard deviations away from the mean

'''


import argparse
import numpy as np
from tqdm import tqdm
import os

from guided_diffusion.image_datasets import (
    ImageDataset, 
    _list_image_files_recursively
)


def main():
    parser = argparse.ArgumentParser(description='From a ds of outliers, create new ds clipped to specified standard deviations away from the mean')
    parser.add_argument('--outlier_dir', type=str,
                    help='directory to pull triplanes from', required=True)
    parser.add_argument('--outdir', type=str,
                    help='where to place dataset', required=True)
    parser.add_argument('--stds', type=float, default=5, required=False,
                    help='allowable stds away from mean before clipping')
    parser.add_argument('--resolution', type=str, default=128, required=False,
                    help='image resolution')
    args = parser.parse_args()

    file_list = _list_image_files_recursively(args.outlier_dir)
    ds = ImageDataset(args.resolution, file_list, normalize=False)

    os.makedirs(args.outdir, exist_ok=True)

    stds = np.load('util/stds.npy')
    means = np.load('util/means.npy')

    lower_bound = means - (args.stds * stds)
    upper_bound = means + (args.stds * stds)

    print(f'lower_bound: {lower_bound}\nupper_bound: {upper_bound}')

    for idx, outlier in tqdm(enumerate(ds)):

        # Clip triplane
        triplane = outlier[0]  # (96, 128, 128)
        clipped_triplane = np.clip(a=triplane, a_min=lower_bound.reshape(-1, 1, 1), a_max=upper_bound.reshape(-1, 1, 1))
        
        # Clipping metrics
        same_values = triplane == clipped_triplane
        same_values = same_values.astype(int)
        print(f'Clipped {100*np.sum(1-same_values)/same_values.size}% of values.')

        np.save(f'{args.outdir}/{os.path.basename(file_list[idx])}', clipped_triplane)


if __name__ == "__main__":
    main()
