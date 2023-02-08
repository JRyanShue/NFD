

'''
Train NeRF auto-decoder on images.

'''

import sys
for path in ['/viscam/u/jrshue/neural-field-diffusion', '/viscam/projects/triplane-diffusion/neural-field-diffusion']:
    if path not in sys.path:
        sys.path.append(path)
    
import os
import argparse
import numpy as np
from PIL import Image
import torch
from triplane_fitting.dataset import SingleSceneNerfDataset, NerfDataset
from triplane_fitting.networks import TriplaneAutoDecoder
from torch.utils.data import DataLoader
# from triplane_fitting.utils.regularization import edr_loss

import time

import importlib
wisp_config_parser = importlib.import_module('modules.kaolin-wisp.wisp.config_parser')
wisp_octree_as = importlib.import_module('modules.kaolin-wisp.wisp.accelstructs.octree_as')
spc_render = importlib.import_module('modules.kaolin.kaolin.render.spc')
wisp_core = importlib.import_module('modules.kaolin-wisp.wisp.core')

from guided_diffusion.script_util import (
    add_dict_to_argparser
)



def main():
    args, args_str = wisp_config_parser.argparse_to_str(create_argparser())

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    timestamp = int(time.time())  # So that we can have multiple runs under the same name
    save_path = f'{args.save_path}/{timestamp}'

    if args.log_online:
        import wandb
        wandb.init(project="nerf-triplane-autodecoder")

    # Create logging directories
    os.makedirs(f'{save_path}/images', exist_ok=True)
    os.makedirs(f'{save_path}/ckpts', exist_ok=True)

    # Deterministic dataloader
    # TODO(JRyanShue): Set sample_points back to True as soon as possible! Will train only on a subset of the data.
    ds = NerfDataset(dataset_path=f'{args.data_dir}', subset_size=args.subset_size, ray_batch_size=args.ray_batch_size, ds_type='train', sample_points=False)  # For now will only use one batch at a time
    ds.init()
    ds.train()

    # dataloader = DataLoader(
    #     ds, 
    #     batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=True, pin_memory=True  # Tune num_workers
    # )

    pipeline, _, _ = wisp_config_parser.get_modules_from_config(args)
    optim_cls, optim_params = wisp_config_parser.get_optimizer_from_config(args)  # Not currently used!
    
    print(f'Pipeline:\n{pipeline}')
    
    how_many_scenes = len(ds)
    auto_decoder = TriplaneAutoDecoder(resolution=args.resolution, channels=args.channels, 
        how_many_scenes=how_many_scenes, input_dim=3, output_dim=4, aggregate_fn=args.aggregate_fn, 
        use_tanh=args.use_tanh, view_embedding=(not args.nouse_view_embed), 
        triplane_cpu_intermediate=args.triplane_cpu_intermediate, device=device)
    print(f'auto_decoder: {auto_decoder}')

    # Link params to optimizer with varying learning rates
    params = [
        {'params': auto_decoder.net.parameters(), 'lr': args.lr},  # 0.001 -> 0.0001?
    ] + [
        {'params': auto_decoder.embeddings[i].parameters(), 'lr': args.lr*args.grid_lr_weight}  # 0.1
        for i in range(how_many_scenes)
        ]
    optimizer = optim_cls(params=params, lr=args.lr)
    # optimizer = torch.optim.Adam(params=params, lr=args.lr, 
    #         betas=(0.9, 0.999))  # Can switch to Adam during tuning

    if args.load_ckpt_path:
        opt_checkpoint = torch.load(args.load_ckpt_path.replace('model', 'opt'))
        # if 'current_embeddings.weight' in checkpoint['model_state_dict'].keys():
        #     del checkpoint['model_state_dict']['current_embeddings.weight']
        # print(f"checkpoint['model_state_dict'].keys(): {checkpoint['model_state_dict'].keys()}")
        # for key in checkpoint['model_state_dict']:
        #     print(checkpoint['model_state_dict'][key].shape)
        # print(f"checkpoint['optimizer_state_dict']['state']: {checkpoint['optimizer_state_dict']['state']}")

        # print(len(checkpoint['optimizer_state_dict']['state']))  # e.g. 8        
        # for idx in checkpoint['optimizer_state_dict']['state']:
        #     print(checkpoint['optimizer_state_dict']['state'][idx]['square_avg'].shape)
        # for group in checkpoint['optimizer_state_dict']['param_groups']:
        #     print(group)
        # for key in checkpoint['optimizer_state_dict']:
        #     print(f"checkpoint['optimizer_state_dict'][{key}].keys(): {checkpoint['optimizer_state_dict'][key].keys()}")

        auto_decoder = torch.load(args.load_ckpt_path)
        # auto_decoder.load_state_dict(checkpoint['model_state_dict'])
        # print(f"checkpoint['optimizer_state_dict']: {checkpoint['optimizer_state_dict']}")
        optimizer.load_state_dict(opt_checkpoint['optimizer_state_dict'])
        epoch = opt_checkpoint['epoch']
        loss = opt_checkpoint['loss']
        print(f'Loaded checkpoint from {args.load_ckpt_path}. Resuming training from epoch {epoch} with loss {loss}...')
    else:
        epoch = 0

    auto_decoder.train()
    raymarcher = wisp_octree_as.OctreeAS()
    raymarcher.init_aabb()  # For now, use a simple AABB raymarcher. 

    N_EPOCHS = 3000000 # A big number -- can easily do early stopping with Ctrl+C. 
    step = 0
    load_start_time = time.time()

    for epoch in range(N_EPOCHS)[epoch:]:
        print(f'EPOCH {epoch}...')
        for (obj_idx, example) in ds:

            start_time = time.time()

            scene_img_gts = example['imgs']  # e.g. [66, 16384, 3]
            scene_rays = example['rays']  # e.g. [66, 16384], or something like [66, 75] if 5k ray samples. But each element is an object!
            
            # Cycle through, one image at a time
            for img_idx in range(scene_img_gts.shape[0]):  # Range used to maintain dimensionality at shape index 0

                loss = 0

                # Maps to GPU
                img_gts = scene_img_gts[img_idx:img_idx+1].to(device)  # e.g. [66, 16384, 3]
                rays = scene_rays[img_idx:img_idx+1].to(device)  # e.g. [66, 16384], or something like [66, 75] with 5k ray samples. But each element is an object!
                
                img_gts = img_gts.reshape(args.batch_size, -1, img_gts.shape[-1])
                rays = wisp_core.Rays.cat(rays_list=rays)  # e.g. [1081344]. Need to use Rays methods because rays isn't a single tensor

                ridx, pidx, samples, depths, deltas, boundary = raymarcher.raymarch(rays, 
                    level=0, num_samples=args.num_steps, raymarch_type=args.raymarch_type)  # e.g. samples: [16359, 256, 3]

                # Get the indices of the ray tensor which correspond to hits. Need for image reconstruction later
                ridx_hit = ridx[spc_render.mark_pack_boundaries(ridx.int())]

                # Compute the color and density for each ray and their samples
                hit_ray_d = rays.dirs.index_select(0, ridx)

                # TODO(JRyanShue): add view direction embedding from Kaolin-Wisp
                samples = samples.reshape(args.batch_size, -1, samples.shape[-2], samples.shape[-1])
                rgba = auto_decoder(obj_idx=obj_idx, coordinates=samples, ray_d=hit_ray_d)
                color, density = rgba['rgb'], rgba['sigma']  # e.g. torch.Size([1, 16359, 256, 3]) torch.Size([1, 16359, 256, 1])
                del ridx, rays

                # Compute optical thickness (for rendering)
                tau = density.reshape(-1, 1) * deltas
                del density, deltas
                ray_colors, transmittance = spc_render.exponential_integration(color.reshape(-1, 3), tau, boundary, exclusive=True)
                
                alpha = spc_render.sum_reduce(transmittance, boundary)

                ray_colors = ray_colors.reshape(args.batch_size, -1, ray_colors.shape[-1])  # e.g. torch.Size([1, 16359, 3])
                alpha = alpha.reshape(args.batch_size, -1, alpha.shape[-1])  # e.g. torch.Size([1, 16359, 1])

                # Populate the background w/ white or black
                if args.bg_color == 'white':
                    rgb = torch.ones(img_gts.shape[0], img_gts.shape[1], 3, device=color.device)
                    color = (1.0-alpha) + alpha * ray_colors
                else:
                    rgb = torch.zeros(img_gts.shape[0], img_gts.shape[1], 3, device=color.device)
                    color = alpha * ray_colors

                rgb = rgb.reshape(-1, rgb.shape[-1])
                rgb[ridx_hit.long()] = color
                rgb = rgb.reshape(args.batch_size, -1, rgb.shape[-1])

                rgb_loss = torch.abs(rgb[..., :3] - img_gts[..., :3])
                
                rgb_loss = rgb_loss.mean()

                loss += rgb_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            # Treat a complete pass through all the images in a scene as one step!
            step += 1


            if not step % args.log_every: 
                print(f'Step {step}: Loss {loss.item()}')
                if args.log_online:
                    wandb.log({"loss": loss.item()})
            if not step % args.val_every:
                print(f'Validating at step {step}, loss {loss.item()}...')

                # TODO(JRyanShue): This needs severe cleaning up.

                # Use unseen validation data
                ds.val()

                for (obj_idx, example) in ds:  # Be careful w/ reshaping

                    scene_img_gts = example['imgs']  # e.g. [66, 16384, 3]
                    scene_rays = example['rays']
                    
                    for img_idx in range(scene_img_gts.shape[0]):

                        # Maps to GPU
                        img_gts = scene_img_gts[img_idx:img_idx+1].to(device)  # e.g. [66, 16384, 3]
                        rays = scene_rays[img_idx:img_idx+1].to(device)

                        img_gts = img_gts.reshape(args.batch_size, -1, img_gts.shape[-1])
                        rays = wisp_core.Rays.cat(rays_list=rays)
                        
                        ridx, pidx, samples, depths, deltas, boundary = raymarcher.raymarch(rays, 
                            level=0, num_samples=args.num_steps, raymarch_type=args.raymarch_type)  # e.g. samples: [16359, 256, 3]
                        ridx_hit = ridx[spc_render.mark_pack_boundaries(ridx.int())]
                        hit_ray_d = rays.dirs.index_select(0, ridx)
                        samples = samples.reshape(args.batch_size, -1, samples.shape[-2], samples.shape[-1])
                        rgba = auto_decoder(obj_idx=obj_idx, coordinates=samples, ray_d=hit_ray_d)
                        color, density = rgba['rgb'], rgba['sigma']  # e.g. torch.Size([1, 16359, 256, 3]) torch.Size([1, 16359, 256, 1])
                        del ridx, rays
                        tau = density.reshape(-1, 1) * deltas
                        del density, deltas
                        ray_colors, transmittance = spc_render.exponential_integration(color.reshape(-1, 3), tau, boundary, exclusive=True)
                        alpha = spc_render.sum_reduce(transmittance, boundary)
                        ray_colors = ray_colors.reshape(args.batch_size, -1, ray_colors.shape[-1])  # e.g. torch.Size([1, 16359, 3])
                        alpha = alpha.reshape(args.batch_size, -1, alpha.shape[-1])  # e.g. torch.Size([1, 16359, 1])
                        # Populate the background w/ white or black
                        if args.bg_color == 'white':
                            rgb = torch.ones(img_gts.shape[0], img_gts.shape[1], 3, device=color.device)
                            color = (1.0-alpha) + alpha * ray_colors
                        else:
                            rgb = torch.zeros(img_gts.shape[0], img_gts.shape[1], 3, device=color.device)
                            color = alpha * ray_colors
                        rgb = rgb.reshape(-1, rgb.shape[-1])
                        rgb[ridx_hit.long()] = color
                        rgb = rgb.reshape(args.batch_size, -1, rgb.shape[-1])

                        rendered_imgs = rgb.detach().cpu().numpy()
                        img_gts = img_gts.detach().cpu().numpy()
                        for img_gt, rendered_img in zip(img_gts, rendered_imgs):
                            # Save rendered image
                            os.makedirs(f'{save_path}/images/step_{step}_loss_{loss.item()}/{obj_idx.item()}', exist_ok=True)
                            img = Image.fromarray((rendered_img * 255).astype(np.uint8).reshape(int(np.sqrt(rendered_img.shape[0])), int(np.sqrt(rendered_img.shape[0])), -1))
                            img.save(f'{save_path}/images/step_{step}_loss_{loss.item()}/{obj_idx.item()}/{img_idx}.png')

                            # Save GT image
                            os.makedirs(f'{save_path}/images/gt/{obj_idx.item()}', exist_ok=True)
                            img = Image.fromarray((img_gt * 255).astype(np.uint8).reshape(int(np.sqrt(img_gt.shape[0])), int(np.sqrt(img_gt.shape[0])), -1))
                            img.save(f'{save_path}/images/gt/{obj_idx.item()}/{img_idx}.png')

                # Revert dataset for continue training
                ds.train()

                if args.log_online:
                    pass  # TODO(JRyanShue): Less of a priority, but should get online logging working with not just loss
            if not step % args.save_every:
                print(f'Saving checkpoint at step {step}...')
                # print(f'auto_decoder.state_dict(): {auto_decoder.state_dict()}')
                # print(f'optimizer.state_dict(): {optimizer.state_dict()}')
                torch.save({
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, f'{save_path}/ckpts/opt_epoch_{epoch}_loss_{loss.item()}.pt')
                torch.save(auto_decoder, f'{save_path}/ckpts/model_epoch_{epoch}_loss_{loss.item()}.pt')
            
            # print(f'Time to do everything besides loading: {time.time() - start_time}')
            load_start_time = time.time()


def create_argparser():
    parser = argparse.ArgumentParser(description='Train NeRF auto-decoder on images.')
    wisp_default_args = dict(activation_type='relu', ao=False, as_type='none', base_lod=5, 
        bg_color='white', camera_clamp=[0, 10], camera_fov=30, camera_lookat=[0, 0, 0], 
        camera_origin=[-3.0, 0.65, -3.0], camera_proj='persp', codebook_bitwidth=8, 
        config='modules/kaolin-wisp/configs/triplanar_nerf.yaml', dataset_num_workers=4, 
        dataset_path='/viscam/projects/triplane-diffusion/data/test_car', 
        dataset_type='multiview', decoder_type='basic', detect_anomaly=False, 
        embedder_type='positional', epochs=50, exp_name='test-triplanar-nerf', 
        feature_bias=0.0, feature_dim=4, feature_std=0.01, get_normals=False, 
        grid_lr_weight=100.0, grid_type='TriplanarGrid', grow_every=-1, 
        growth_strategy='increase', hidden_dim=128, interpolation_type='linear', layer_type='none', 
        log_2d=False, log_dir='_results/logs/runs/', log_level=20, lr=0.001, 
        matcap_path='data/matcaps/matcap_plastic_yellow.jpg', max_grid_res=2048, min_dis=0.0003, 
        mip=0, mode_mesh_norm='sphere', model_format='full', multiscale_type='sum', 
        multiview_dataset_format='rtmv', nef_type='NeuralRadianceField', noise_std=0.0, 
        num_layers=1, num_lods=4, num_rays_sampled_per_img=4096, num_samples=100000, 
        num_samples_on_mesh=100000000, num_steps=256, only_last=False, optimizer_type='rmsprop', 
        out_dim=4, perf=False, pos_multires=10, position_input=False, pretrained=None, prune_every=-1, 
        random_lod=False, raymarch_type='voxel', render_batch=4000, render_every=10, 
        render_res=[1024, 1024], resample=False, resample_every=1, rgb_loss=1.0, 
        sample_mode=['rand', 'near', 'near', 'trace', 'trace'], sample_tex=False, samples_per_voxel=256, 
        save_as_new=False, shading_mode='rb', shadow=False, skip=None, step_size=1.0, 
        tracer_type='PackedRFTracer', trainer_type='MultiviewTrainer', tree_type='quad', valid_every=50, 
        valid_only=False, view_multires=4, weight_decay=0
    )
    add_dict_to_argparser(parser, wisp_default_args)
    parser.add_argument('--data_dir', type=str,
                    help='directory to pull data from', required=True)
    parser.add_argument('--batch_size', type=int, default=1, required=False,
                    help='number of scenes per batch')
    parser.add_argument('--ray_batch_size', type=int, default=4000, required=False,
                    help='number of rays per scene per batch to train volumetric rendering on')
    parser.add_argument('--log_every', type=int, default=20, required=False)
    parser.add_argument('--val_every', type=int, default=100, required=False)
    parser.add_argument('--save_every', type=int, default=200, required=False)
    parser.add_argument('--load_ckpt_path', type=str, default=None, required=False,
                    help='checkpoint to continue training from')
    parser.add_argument('--save_path', type=str, default='./ckpts', required=False,
                    help='where to save model checkpoints as well as validation stuff')
    parser.add_argument('--resolution', type=int, default=128, required=False,
                    help='triplane resolution')
    parser.add_argument('--channels', type=int, default=4, required=False,
                    help='triplane depth')
    parser.add_argument('--aggregate_fn', type=str, default='sum', required=False,
                    help='function for aggregating triplane features')
    parser.add_argument('--subset_size', type=int, default=None, required=False,
                    help='size of the dataset subset if we\'re training on a subset')
    parser.add_argument('--steps_per_batch', type=int, default=1, required=False,
                    help='If specified, how many GD steps to run on a batch before moving on. To address I/O stuff.')
    parser.add_argument('--edr_val', type=float, default=None, required=False,
                    help='If specified, use explicit density regularization with the specified offset distance value.')
    parser.add_argument('--use_tanh', default=False, required=False, action='store_true',
                    help='Whether to use tanh to clamp triplanes to [-1, 1].')
    parser.add_argument('--nouse_view_embed', default=False, required=False, action='store_true',
                    help='Whether to concat view embedding with feature embedding input into net.')
    parser.add_argument('--log_online', default=False, required=False, action='store_true',
                    help='Whether to log with wandb.')
    parser.add_argument('--triplane_cpu_intermediate', default=False, required=False, action='store_true',
                    help='Whether to store triplane parameters on CPU until needed and shifted to GPU.')            
    return parser


if __name__ == "__main__":
    main()

