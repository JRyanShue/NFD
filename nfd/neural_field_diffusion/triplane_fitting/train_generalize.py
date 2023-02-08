'''
Freeze the pretrained decoder and optimize triplanes to new shapes to test the generality of the MLP.

e.g. 
CUDA_VISIBLE_DEVICES=8 python triplane_diffusion/train_generalize.py --data_dir \
../data/preprocessed/02958343 --batch_size 10 --load_ckpt_path \
/home/jrshue/neural-field-diffusion/ckpts_500shapeAD_edr0.01_128res/model_epoch_1999_loss_0.19445328414440155.pt \
--checkpoint_path ckpts_triplanes_700-800_500to100_shape_edr0.01_128res --training_subset_size 500 \
--skip_number 200 --subset_size 100 --edr_val 0.01 --save_every 2000
'''


import sys
if '/viscam/u/jrshue/neural-field-diffusion' not in sys.path:
    sys.path.append('/viscam/u/jrshue/neural-field-diffusion')
    
import os
import argparse
import numpy as np
import torch
from dataset import OccupancyDataset, SingleExampleOccupancyDataset, BigTensorOccupancyDataset, GeneralizeDataset
from torch.utils.data import DataLoader
from networks import CartesianPlaneNonSirenEmbeddingNetwork, TriplaneAutoDecoder
from utils.regularization import edr_loss
import wandb
import time


def main():
    parser = argparse.ArgumentParser(description='Freeze the pretrained decoder and optimize triplanes to new shapes to test the generality of the MLP.')
    parser.add_argument('--data_dir', type=str,
                    help='directory to pull data from', required=True)
    parser.add_argument('--batch_size', type=int, default=1, required=False,
                    help='number of scenes per batch')
    parser.add_argument('--points_batch_size', type=int, default=500000, required=False,
                    help='number of points per scene, nss and uniform combined')
    parser.add_argument('--log_every', type=int, default=20, required=False)
    parser.add_argument('--val_every', type=int, default=100, required=False)
    parser.add_argument('--save_every', type=int, default=200, required=False)
    parser.add_argument('--load_ckpt_path', type=str, required=True,
                    help='checkpoint to continue training from')
    parser.add_argument('--checkpoint_path', type=str, default='./ckpts', required=False,
                    help='where to save model checkpoints')
    parser.add_argument('--resolution', type=int, default=128, required=False,
                    help='triplane resolution')
    parser.add_argument('--channels', type=int, default=32, required=False,
                    help='triplane depth')
    parser.add_argument('--aggregate_fn', type=str, default='sum', required=False,
                    help='function for aggregating triplane features')
    parser.add_argument('--training_subset_size', type=int, default=500, required=True,
                    help='size of the dataset subset we trained on, to skip seen examples.')
    parser.add_argument('--skip_number', type=int, default=0, required=False,
                    help='How many extra examples to skip, if we are training triplanes in parallel across multiple machines.')
    parser.add_argument('--subset_size', type=int, default=1, required=True,
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

    wandb.init(project="triplane-autodecoder")

    os.makedirs(args.checkpoint_path, exist_ok=True)

    # Deterministic dataloader
    dataloader = DataLoader(
        GeneralizeDataset(dataset_path=args.data_dir, points_batch_size=args.points_batch_size, training_subset_size=args.training_subset_size + args.skip_number, subset_size=args.subset_size), 
        batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=True, pin_memory=True  # Tune num_workers
    )

    # Triplane auto-decoder
    # Start with train settings then switch to generalization settings
    how_many_scenes = args.training_subset_size
    auto_decoder = TriplaneAutoDecoder(resolution=args.resolution, channels=args.channels, how_many_scenes=how_many_scenes, input_dim=3, output_dim=1, aggregate_fn=args.aggregate_fn, use_tanh=args.use_tanh).to(device)
    
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Load whole model, then replace embeddings
    if args.load_ckpt_path:
        checkpoint = torch.load(args.load_ckpt_path)
        auto_decoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f'Loaded checkpoint from {args.load_ckpt_path}. Training by optimizing triplanes only...')

    # Swap in new embeddings.
    how_many_scenes = args.subset_size
    auto_decoder.embeddings = TriplaneAutoDecoder(resolution=args.resolution, channels=args.channels, how_many_scenes=how_many_scenes, input_dim=3, output_dim=1, aggregate_fn=args.aggregate_fn).to(device).embeddings
    # Only optimize triplanes (embedding layer)!!
    optimizer = torch.optim.Adam(params=auto_decoder.embeddings.parameters(), lr=1e-3, 
            betas=(0.9, 0.999))
    print(auto_decoder)
    auto_decoder.embeddings.train()

    N_EPOCHS = 3000000  # A big number -- can easily do early stopping with Ctrl+C. 
    step = 0
    load_start_time = time.time()
    for epoch in range(N_EPOCHS):
        print(f'EPOCH {epoch}...')
        for (obj_idx, pts_sdf) in dataloader:
            
            start_time = time.time()

            obj_idx, pts_sdf = obj_idx.int().to(device), pts_sdf.float().to(device)

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
            
            load_start_time = time.time()




if __name__ == "__main__":
    main()

