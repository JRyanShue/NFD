

'''
Sweep across all checkpoints in given range, logging metrics.
For example, we want to look for the checkpoint with the highest SD.

'''

from guided_diffusion.image_datasets import (
    ImageDataset, 
    _list_image_files_recursively,
    load_data
)
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from scripts.image_train import create_argparser
from get_stats import (
    recursive_combine_variance
)

import argparse
import numpy as np
from tqdm import tqdm

import sys
print(sys.path)
sys.path.append('/home/jrshue/neural-field-diffusion')
import os

from mpi4py import MPI

import torch as th


n_per_channel = 128 * 128
num_channels = 96
num_examples = 15001

num_stds = 5


def main():
    parser = create_argparser()
    parser.add_argument('--resolution', type=str, default=128, required=False,
                    help='image resolution')
    parser.add_argument('--ckpt_dir', type=str,
                    help='directory to pull checkpoints from', required=True)
    parser.add_argument('--ckpt_range', type=str,
                    help='range of checkpoint step #s', required=True)
    parser.add_argument('--break_early', type=bool, default=True,
                    help='just get the initial metrics', required=False)
    parser.add_argument('--ckpt_step', type=int, default=2500,
                    help='distance in steps between adjacent checkpoints', required=False)
    args = parser.parse_args()

    print(f'Pulling checkpoints from {args.ckpt_dir}...')

    # Get list of checkpoints to try to pull
    lower_bound, upper_bound = [int(x) for x in args.ckpt_range.split('-')]
    checkpoint_steps = [args.ckpt_step * x + lower_bound for x in range(int((upper_bound - lower_bound) / args.ckpt_step) + 1)]

    def filter_func(file):
        if all(x in file for x in ['model', '.pt']) and int(file.split('.')[0][5:]) in checkpoint_steps:
            return True
        else:
            return False
    checkpoint_files = list(filter(filter_func, os.listdir(args.ckpt_dir)))

    print(f'Evaluating the following checkpoints: {checkpoint_files}...')
    
    # Log hyperparameter configuration
    print(f'hparam dict: {args_to_dict(args, model_and_diffusion_defaults().keys())}')

    # We're going to write everything from the different checkpoints to the same log
    dist_util.setup_dist()
    logger.configure()


    # Loop through all checkpoints to obtain metrics.
    for file in checkpoint_files:

        ckpt_path = f'{args.ckpt_dir}/{file}'

        logger.log(f"creating model and diffusion with ckpt from {ckpt_path}...")

        # Set checkpoint to use
        args.resume_checkpoint = ckpt_path

        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        print(f'dist_util.dev() {dist_util.dev()}')
        model.to(dist_util.dev())
        # print(f'model.device: {model.device}')
        print(f'Number of devices: {th.cuda.device_count()}')
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

        logger.log("creating data loader...")
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
        )

        logger.log("training...")
        TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=args.batch_size,
            microbatch=args.microbatch,
            lr=args.lr,
            ema_rate=args.ema_rate,
            log_interval=args.log_interval,
            val_interval=args.val_interval,
            save_interval=args.save_interval,
            resume_checkpoint=args.resume_checkpoint,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.lr_anneal_steps,
            wandb=None,
            break_early=args.break_early,
        ).run_loop()

if __name__ == "__main__":
    main()
