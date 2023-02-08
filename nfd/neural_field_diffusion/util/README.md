
# Utils for explicit triplane normalization

To generate a set of stats for a given dataset:

    python get_stats.py --data_dir [triplane directory] --save_dir [stats directory, e.g. ./util/stats_dir]

It's important to specify --save_dir. When training the diffusion model, this directory is passed as --stats_dir.

The script will loop through the triplane directory (pretty quickly) and find the channel-wise mean and SD. 
Will shift the triplanes to be centered at 0 and clip after a certain multiple of the SD.
The distribution of the triplanes with the L2 regularization is already pretty good, so the clipping part only clips values for a few extreme outliers. The SD multiple is set in line 23 of get_stats.py

Some experiments:
1. What happens with a lower clipping threshold (clipping more outliers)? If this doesn't impact performance, might be worth it. Though the L2 regularization should already be normalizing things enough.