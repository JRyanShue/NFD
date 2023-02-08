

'''
Train NeRF on single scene.

'''

import sys
if '/viscam/u/jrshue/neural-field-diffusion' not in sys.path:
    sys.path.append('/viscam/u/jrshue/neural-field-diffusion')
    
import os
import argparse
import numpy as np
import torch

from triplane_fitting.dataset import SingleSceneNerfDataset
from modules.kaolin-wisp.wisp import 
from torch.utils.data import DataLoader
from triplane_fitting.networks import CartesianPlaneNonSirenEmbeddingNetwork, TriplaneAutoDecoder
from triplane_fitting.utils.regularization import edr_loss
import wandb
import time



def main():
    parser = argparse.ArgumentParser(description='Train NeRF auto-decoder on images.')
    parser.add_argument('--data_dir', type=str,
                    help='directory to pull data from', required=True)
    parser.add_argument('--batch_size', type=int, default=1, required=False,
                    help='number of scenes per batch')
    parser.add_argument('--points_batch_size', type=int, default=500000, required=False,
                    help='number of points per scene, nss and uniform combined')
    parser.add_argument('--log_every', type=int, default=20, required=False)
    parser.add_argument('--val_every', type=int, default=100, required=False)
    parser.add_argument('--save_every', type=int, default=200, required=False)
    parser.add_argument('--load_ckpt_path', type=str, default=None, required=False,
                    help='checkpoint to continue training from')
    parser.add_argument('--checkpoint_path', type=str, default='./ckpts', required=False,
                    help='where to save model checkpoints')
    parser.add_argument('--resolution', type=int, default=128, required=False,
                    help='triplane resolution')
    parser.add_argument('--channels', type=int, default=32, required=False,
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
    args = parser.parse_args()

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    wandb.init(project="nerf-triplane-autodecoder")

    os.makedirs(args.checkpoint_path, exist_ok=True)

    # When you load the entire dataset onto GPU memory...
    # gpu_dataset = BigTensorOccupancyDataset(dataset_path=args.data_dir, batch_size=args.batch_size, points_batch_size=args.points_batch_size, subset_size=args.subset_size, device=device)

    # Deterministic dataloader
    dataloader = DataLoader(
        OccupancyDataset(dataset_path=args.data_dir, points_batch_size=args.points_batch_size, subset_size=args.subset_size), 
        batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=True, pin_memory=True  # Tune num_workers
    )
    # dataloader = DataLoader(
    #     SingleExampleOccupancyDataset(dataset_path=args.data_dir), 
    #     batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True  # Tune num_workers
    # )

    # Triplane auto-decoder
    # how_many_scenes = len(dataloader) * args.batch_size
    how_many_scenes = len(gpu_dataset)
    # Output dim = 4!
    auto_decoder = TriplaneAutoDecoder(resolution=args.resolution, channels=args.channels, how_many_scenes=how_many_scenes, input_dim=3, output_dim=4, aggregate_fn=args.aggregate_fn, use_tanh=args.use_tanh).to(device)
    
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adam(params=auto_decoder.parameters(), lr=1e-5, 
    #         betas=(0.9, 0.999))
    optimizer = torch.optim.Adam(params=auto_decoder.parameters(), lr=1e-3, 
            betas=(0.9, 0.999))

    if args.load_ckpt_path:
        checkpoint = torch.load(args.load_ckpt_path)
        auto_decoder.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f'Loaded checkpoint from {args.load_ckpt_path}. Resuming training from epoch {epoch} with loss {loss}...')

    auto_decoder.train()

    N_EPOCHS = 3000000 # A big number -- can easily do early stopping with Ctrl+C. 
    step = 0
    load_start_time = time.time()
    for epoch in range(N_EPOCHS):
        print(f'EPOCH {epoch}...')
        # for (obj_idx, pts_sdf) in dataloader:
        # for (coordinates, gt_occupancies) in dataloader:
        idx_array = torch.Tensor(np.array(list(range(len(gpu_dataset))))).long().to(device)
        idx_array = idx_array.reshape(-1, args.batch_size)
        for obj_idx in idx_array:
            # print(f'Time to load data with CPU: {time.time() - load_start_time}')
            start_time = time.time()

            obj_idx, pts_sdf = obj_idx.int().to(device), gpu_dataset[obj_idx]
            # print(obj_idx, pts_sdf.shape)
            # pts_sdf = pts_sdf.float().to(device)
            # print(f'Time to move data from CPU to GPU: {time.time() - start_time}')

            # # Sample on GPU
            # sample_indices = torch.Tensor(args.batch_size, args.points_batch_size).uniform_(0, pts_sdf.shape[1]).long().to(device)  # .expand_as(pts_sdf)
            # # TODO @JRyanShue or @zankner: use torch.gather() or a better function for this to get rid of for loop. Though it only loops over batch_size so it's not too bad.
            # sampled_pts_sdf = torch.cat([batch_elem[index_elem].unsqueeze(0) for batch_elem, index_elem in zip(pts_sdf, sample_indices)])
            
            # coordinates, gt_occupancies = sampled_pts_sdf[..., 0:3], sampled_pts_sdf[..., -1]
            coordinates, gt_occupancies = pts_sdf[..., 0:3], pts_sdf[..., -1]

            start_forward_backward = time.time()
            
            for _step in range(args.steps_per_batch):                
                pred_occupancies = auto_decoder(obj_idx, coordinates)

                # BCE loss
                loss = loss_fn(pred_occupancies, gt_occupancies.reshape((gt_occupancies.shape[0], gt_occupancies.shape[1], -1)))

                # Explicit density regulation
                if args.edr_val is not None:
                    loss += edr_loss(obj_idx, auto_decoder, device, offset_distance=args.edr_val)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step += 1

            # print(f'Step: {optimizer.state[optimizer.param_groups[0]["params"][-1]]["step"]}, time for all forward passes in batch: {time.time() - start_forward_backward}, ')  # Need step to force async
            
            if not step % args.log_every: 
                print(f'Step {step}: Loss {loss.item()}')
                wandb.log({"loss": loss.item()})
            # if not step % args.val_every:
            #     print(f'Step {step}: Loss {loss.item()}')
            #     wandb.log({"loss": loss.item()})
            if not step % args.save_every:
                print(f'Saving checkpoint at step {step}')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': auto_decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, f'{args.checkpoint_path}/model_epoch_{epoch}_loss_{loss.item()}.pt')
            
            # print(f'Time to do everything besides loading: {time.time() - start_time}')
            load_start_time = time.time()




if __name__ == "__main__":
    main()

