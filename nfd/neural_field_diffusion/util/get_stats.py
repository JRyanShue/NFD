
'''
Normalize the triplane dataset, scaling each channel individually to [-1, 1]

Different channels have vastly different ranges of values, unlike RGB images.
'''

from guided_diffusion.image_datasets import (
    ImageDataset, 
    _list_image_files_recursively
)
import argparse
import numpy as np
from tqdm import tqdm
import os


n_per_channel = 128 * 128
num_channels = 96
num_examples = 15001

# num_stds = 5
num_stds = 16


# Combine variance of two sets by getting their variance if concatenated
def combine_variance(mean_1, mean_2, var_1, var_2, n_per_channel):  # var_1 ex. shape: (num_channels,)
  n_1 = n_per_channel
  n_2 = n_per_channel
  return(
      ( ((n_1 - 1) * (var_1 + var_2)) /  # (num_channels,) /
      (n_1 * 2 - 1) ) +  # (1,)
      ( ((n_1 ** 2) * ((mean_1 - mean_2) ** 2)) /  # (num_channels,) /
      ((n_1 + n_2) * (n_1 + n_2 - 1)) )  # (1,)
  )  # output: (num_channels,)


def combine_means(mean_1, mean_2):
    return (mean_1 + mean_2) / 2


# Collapse the first dimension of array, using it to combine variance
def recursive_combine_variance(mean_arr, var_arr, n_per_channel):

    new_mean_array = np.full((mean_arr.shape[0] // 2, num_channels), 0)
    new_var_array = np.full((var_arr.shape[0] // 2, num_channels), 0)  # Arrays half the size for after collapsing
    
    for idx, mean_and_var in enumerate(zip(mean_arr, var_arr)):
        mean, variance = mean_and_var
        if idx % 2:  # Every second, collapse variance with the value before and add to new array
            new_mean_array[idx//2] = combine_means(mean_1=mean_arr[idx-1], mean_2=mean)
            new_var_array[idx//2] = combine_variance(mean_1=mean_arr[idx-1], mean_2=mean, var_1=var_arr[idx-1], var_2=variance, n_per_channel=n_per_channel)
        # print(f'{idx}: mean: {mean}; variance: {variance}'
    
    # Recursion process
    if new_var_array.shape[0] == 1:
        return new_var_array
    else:
        # We have double the samples represented per channel in the next pass.
        return(recursive_combine_variance(mean_arr=new_mean_array, var_arr=new_var_array, n_per_channel=n_per_channel*2))


def main():
    parser = argparse.ArgumentParser(description='Normalize a dataset of triplanes')
    parser.add_argument('--resolution', type=str, default=128, required=False,
                    help='image resolution')
    parser.add_argument('--data_dir', type=str,
                    help='directory to pull triplanes from', required=True)
    parser.add_argument('--save_dir', type=str, default='./util',
                    help='where to save stats.', required=False)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    
    print(args.resolution, args.data_dir)
    ds = ImageDataset(args.resolution, sorted(_list_image_files_recursively(args.data_dir)), normalize=False)
    num_examples = len(ds)
    print(f'Finding statistics across {num_examples} examples.')

    min_values = np.full((num_channels), 1000)  # min for each channel
    max_values = np.full((num_channels), -1000)  # max for each channel
    mean_values = np.full((num_examples, num_channels), 0.0)  # mean for each channel in the ds (to be filled)
    var_values = np.full((num_examples, num_channels), 0.0)  # variance for each channel

    idx = None
    # for idx, triplane in tqdm(enumerate(ds)):
    #     triplane = triplane[0]  # (96, 128, 128)
    #     _min = np.amin(triplane, axis=(1, 2))  # Collapse two dimensions
    #     _max = np.amax(triplane, axis=(1, 2))  # Collapse two dimensions

    #     # Update min and max
    #     min_values = np.where(_min < min_values, _min, min_values)
    #     max_values = np.where(_max > max_values, _max, max_values)

    #     # Update mean and SD
    #     mean_values[idx] = np.mean(triplane, axis=(1, 2))
    #     var_values[idx] = np.var(triplane, axis=(1, 2))

    # One big tensor method:
    ds_tensor = np.full((num_examples, num_channels, 128, 128), 0.0)
    for idx, triplane in tqdm(enumerate(ds)):
        triplane = triplane[0]
        ds_tensor[idx] = triplane
    ds_tensor = np.transpose(ds_tensor, axes=[1, 0, 2, 3])
    min_values = np.amin(ds_tensor, axis=(1, 2, 3))
    max_values = np.amax(ds_tensor, axis=(1, 2, 3))
    means = np.mean(ds_tensor, axis=(1, 2, 3))
    variances = np.var(ds_tensor, axis=(1, 2, 3))

    print(f'idx: {idx}')

    # Calculate mean
    # means = np.mean(mean_values, axis=0)

    # Calculate variances & SD for each channel
    # print(f'var_values[0]: {var_values[0]}')
    
    # n_per_channel_global = num_examples * n_per_channel  # Total number of values being combined to generate a statistic for a specific channel.
    # variances = recursive_combine_variance(mean_values, var_values, n_per_channel)
    stds = np.sqrt(variances)

    print(f'mean: {np.mean(means)}')
    print(f'mean SD: {np.mean(stds)}')

    np.save(f'{args.save_dir}/min_values', min_values)
    np.save(f'{args.save_dir}/max_values', max_values)
    np.save(f'{args.save_dir}/variances', variances)
    np.save(f'{args.save_dir}/stds', stds)
    np.save(f'{args.save_dir}/means', means)
    
    lower_bound = means - (num_stds * stds)
    upper_bound = means + (num_stds * stds)

    np.save(f'{args.save_dir}/lower_bound', lower_bound)
    np.save(f'{args.save_dir}/upper_bound', upper_bound)

    combined_arr = np.concatenate((min_values.reshape(-1, 1), max_values.reshape(-1, 1), means.reshape(-1, 1), variances.reshape(-1, 1)), axis=1)
    
    # Save csv
    np.savetxt(f'{args.save_dir}/stats.csv', combined_arr, delimiter=',')

    for idx, line in enumerate(combined_arr):
        print(f'{idx}: min: {line[0]}; max: {line[1]}; mean: {line[2]}; variance: {line[3]}; SD: {np.sqrt(line[3])}')
    # print(f'min_values: {min_values}\nmax_values: {max_values}')


if __name__ == "__main__":
    main()
