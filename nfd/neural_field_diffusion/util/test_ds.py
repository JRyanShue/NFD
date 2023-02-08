
'''
Loop through the dataset, with normalization enabled, to see how the ranges function.

'''

from guided_diffusion.image_datasets import (
    ImageDataset, 
    _list_image_files_recursively
)
from get_stats import (
    recursive_combine_variance
)
import argparse
import numpy as np
from tqdm import tqdm


n_per_channel = 128 * 128
num_channels = 96
num_examples = 15001

num_stds = 5



def main():
    parser = argparse.ArgumentParser(description='Test a dataset of triplanes')
    parser.add_argument('--resolution', type=str, default=128, required=False,
                    help='image resolution')
    parser.add_argument('--data_dir', type=str,
                    help='directory to pull triplanes from', required=True)
    args = parser.parse_args()
    
    # !!!!!!!! NORMALIZE=TRUE
    ds = ImageDataset(args.resolution, _list_image_files_recursively(args.data_dir), normalize=True)

    min_values = np.full((num_channels), 1000.0)  # min for each channel
    max_values = np.full((num_channels), -1000.0)  # max for each channel
    mean_values = np.full((num_examples, num_channels), 0.0)  # mean for each channel in the ds (to be filled)
    var_values = np.full((num_examples, num_channels), 0.0)  # variance for each channel

    idx = None
    for idx, triplane in tqdm(enumerate(ds)):
        triplane = triplane[0]  # (96, 128, 128)
        _min = np.amin(triplane, axis=(1, 2))  # Collapse two dimensions
        _max = np.amax(triplane, axis=(1, 2))  # Collapse two dimensions

        # Update min and max
        min_values = np.where(_min < min_values, _min, min_values)
        max_values = np.where(_max > max_values, _max, max_values)

        # Update mean and SD
        # print(np.mean(triplane, axis=(1, 2)), mean_values.dtype)
        mean_values[idx] = np.mean(triplane, axis=(1, 2))
        var_values[idx] = np.var(triplane, axis=(1, 2))
        # break

    print(f'idx: {idx}')

    # Calculate mean
    means = np.mean(mean_values, axis=0)

    # Calculate variances & SD for each channel
    
    n_per_channel_global = num_examples * n_per_channel  # Total number of values being combined to generate a statistic for a specific channel.
    variances = recursive_combine_variance(mean_values, var_values, n_per_channel)
    stds = np.sqrt(variances)

    # np.save('util/min_values', min_values)
    # np.save('util/max_values', max_values)
    # np.save('util/variances', variances)
    # np.save('util/stds', stds)
    # np.save('util/means', means)
    
    # lower_bound = means - (num_stds * stds)
    # upper_bound = means + (num_stds * stds)

    # np.save('util/lower_bound', lower_bound)
    # np.save('util/upper_bound', upper_bound)

    combined_arr = np.concatenate((min_values.reshape(-1, 1), max_values.reshape(-1, 1), means.reshape(-1, 1), variances.reshape(-1, 1), stds.reshape(-1, 1)), axis=1)
    
    # Save csv
    # np.savetxt('util/stats.csv', combined_arr, delimiter=',')

    for idx, line in enumerate(combined_arr):
        print(f'{idx}: min: {line[0]}; max: {line[1]}; mean: {line[2]}; variance: {line[3]}; SD: {line[4]}')
    # print(f'min_values: {min_values}\nmax_values: {max_values}')


if __name__ == "__main__":
    main()
