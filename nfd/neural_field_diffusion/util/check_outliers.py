
import argparse
import numpy as np
from tqdm import tqdm

from guided_diffusion.image_datasets import (
    ImageDataset, 
    _list_image_files_recursively
)

'''
Check through dataset to see if values go past a certain amount of SD's

Report outliers.
'''


def main():
    parser = argparse.ArgumentParser(description='Find outliers.')
    parser.add_argument('--data_dir', type=str,
                    help='directory to pull triplanes from', required=True)
    parser.add_argument('--resolution', type=str, default=128, required=False,
                    help='image resolution')
    parser.add_argument('--stds', type=int, default=2.5, required=False,
                    help='allowable stds before reporting outlier')
    parser.add_argument('--stats_dir', type=str, default='util', required=False,
                    help='where to pull stats from.')
    args = parser.parse_args()

    ds = ImageDataset(args.resolution, _list_image_files_recursively(args.data_dir), normalize=False)

    stds = np.load(f'{args.stats_dir}/stds.npy')
    means = np.load(f'{args.stats_dir}/means.npy')

    print(f'stds: {stds}')
    print(f'means: {means}')

    lower_bound = means - (args.stds * stds)
    upper_bound = means + (args.stds * stds)

    print(f'lower_bound: {lower_bound}')
    print(f'upper_bound {upper_bound}')

    print(np.concatenate((lower_bound.reshape(-1, 1), upper_bound.reshape(-1, 1)), axis=1))

    for idx, triplane in tqdm(enumerate(ds)):
        triplane = triplane[0]  # (96, 128, 128)
        _min = np.amin(triplane, axis=(1, 2))  # Collapse two dimensions -> (96,)
        _max = np.amax(triplane, axis=(1, 2))
        if (_min < lower_bound).any() or (_max > upper_bound).any():
            print(f'OUTLIER: {idx}')



if __name__ == "__main__":
    main()
