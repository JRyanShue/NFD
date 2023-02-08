
'''
From a list of outliers, create an outliers dataset for testing

'''


import argparse
import numpy as np
from tqdm import tqdm
import os

from guided_diffusion.image_datasets import (
    ImageDataset, 
    _list_image_files_recursively
)

outliers = [
    89, 2242, 4650, 4949, 4956, 6206, 6595, 6596, 7343, 9082, 9743, 11247
]


def main():
    parser = argparse.ArgumentParser(description='Make dataset of outliers.')
    parser.add_argument('--data_dir', type=str,
                    help='directory to pull triplanes from', required=True)
    parser.add_argument('--outdir', type=str,
                    help='where to place dataset', required=True)
    parser.add_argument('--resolution', type=str, default=128, required=False,
                    help='image resolution')
    args = parser.parse_args()

    ds = ImageDataset(args.resolution, _list_image_files_recursively(args.data_dir), normalize=False)

    os.makedirs(args.outdir, exist_ok=True)

    for outlier in outliers:
        triplane = ds[outlier][0]
        # Huge ranges
        # print(np.concatenate((np.amin(triplane, axis=(1, 2)).reshape(-1, 1), np.amax(triplane, axis=(1, 2)).reshape(-1, 1)), axis=1))
        print(f'{args.outdir}/{outlier:04d}.npy')
        os.system(f'cp {args.data_dir}/{outlier:04d}.npy {args.outdir}/')


if __name__ == "__main__":
    main()
