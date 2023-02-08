#!/usr/bin/python

"""
Train a diffusion model on images.
"""

import sys
print(sys.path)
sys.path.append('/viscam/projects/triplane-diffusion/neural-field-diffusion')
# sys.path.append('/home/jrshue/neural-field-diffusion')
# sys.path.append('/home/jrshue/miniconda3/envs/nfd/lib/python3.10/site-packages')

import os

import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from mpi4py import MPI

import torch as th

import wandb

if MPI.COMM_WORLD.Get_rank() == 0:
    wandb.init(project="nfd", entity="implicit-mt")


def main(checkpoint=None, in_out_channels=None, break_early=False):
    args = create_argparser().parse_args()

    if checkpoint:
        args.resume_checkpoint = checkpoint
    if in_out_channels: 
        args.in_out_channels = in_out_channels

    # Log hyperparameter configuration
    print(f'hparam dict: {args_to_dict(args, model_and_diffusion_defaults().keys())}')

    if MPI.COMM_WORLD.Get_rank() == 0:
        wandb.config = args_to_dict(args, model_and_diffusion_defaults().keys())

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
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
        explicit_normalization=args.explicit_normalization,
        stats_dir=args.stats_dir,
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
        wandb=wandb,
        explicit_normalization=args.explicit_normalization,
        stats_dir=args.stats_dir,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        val_interval=500,
        save_interval=1000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        local_rank=None,
        explicit_normalization=False,
        stats_dir=None,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
