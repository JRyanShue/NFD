


'''
Make dummy dataset to test I/O speed on different machines. 306MB/scene with 10M points per example

'''

import argparse
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Make dummy dataset to test I/O speed on different machines.')
    parser.add_argument('--dir', type=str,
                    help='directory to make dummy dataset in', required=True)
    parser.add_argument('--points_per_example', type=int, default=10000000, required=False,
                    help='number of points per scene, "nss and uniform combined"')
    parser.add_argument('--size', type=int, default=None, required=True,
                    help='number of examples.')
    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)

    for idx in range(args.size):
        dummy_arr = np.random.rand(args.points_per_example, 4)
        np.save(f'{args.dir}/{idx}', dummy_arr)
        print(f'Made dummy data at idx {idx}')



if __name__ == "__main__":
    main()

