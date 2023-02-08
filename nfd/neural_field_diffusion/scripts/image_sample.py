"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

import sys
from neural_field_diffusion.guided_diffusion import dist_util, logger
from neural_field_diffusion.guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from neural_field_diffusion.util.normalization import unnormalize


def main(args=None):
    if args is not None:
        print('Using args passed via Python')
        args = args
    else:
        args = create_argparser().parse_args()

    print(f'Using stats in {args.stats_dir}...')

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    obj_idx = 0
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        print(f'args.clip_denoised: {args.clip_denoised}')
        print(f'model_kwargs: {model_kwargs}')
        sample = sample_fn(
            model,
            (args.batch_size, 96, args.image_size, args.image_size),  # CONTROL SHAPE HERE
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            save_intermediate=args.save_intermediate,  # Whether to save intermediate noise
            save_timestep_interval=args.save_timestep_interval
        )
        if args.save_intermediate:
            sample, prev_steps = sample

        # RESCALE IMAGE -- Needs to be aligned with input normalization!
        if args.explicit_normalization:
            sample = unnormalize(sample, stats_dir=args.stats_dir)
            if args.save_intermediate:
                for key in prev_steps:
                    prev_steps[key] = unnormalize(prev_steps[key], stats_dir=args.stats_dir)

        # TEST
        # rnt = th.Tensor(np.load('random_normalized_triplane')).to(dist_util.dev())
        # rnt = rnt * (_range / 2) + middle
        # rnt = rnt.permute(0, 2, 3, 1)
        # rnt = rnt.contiguous()
        # np.save('random_unnormalized_triplane', rnt.cpu().numpy())

        # sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        if args.save_intermediate:
            os.makedirs(f'{args.save_dir}/intermediate_tensors', exist_ok=True)
            for key in prev_steps:
                # prev_steps[key] = prev_steps[key].permute(0, 2, 3, 1)
                # TODO(JRyanShue): Fix this beacuse it's janky. Saves in a different shape than how it saves the usual samples. Should permute it back in an external script.
                prev_steps[key] = prev_steps[key].reshape(prev_steps[key].shape[1:])
                prev_steps[key] = prev_steps[key].contiguous()
                intermediate_tensor = prev_steps[key].cpu().numpy()
                print(f'intermediate_tensor.shape: {intermediate_tensor.shape}')
                np.save(f'{args.save_dir}/intermediate_tensors/{obj_idx}_it{key}', intermediate_tensor)

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL  # Gathers across devices
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        if args.save_dir is not None:
            out_path = os.path.join(args.save_dir, f"samples_{shape_str}.npz")
        else:
            out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")

    return arr


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        stats_dir='util/po_triplanes_v2',
        explicit_normalization=False,
        save_dir=None,
        save_intermediate=False,
        save_timestep_interval=20
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
