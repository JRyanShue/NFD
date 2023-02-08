
'''
Use the same dataset from training, evaluating the model's representation of its shapes 
and comparing to the ground truth.

Make sure the specified data directory is the val set, aka a shifted set of points of the same 
shapes that will evaluate the model's true metrics (not its overfit performance).

'''


import sys
if '/viscam/u/jrshue/neural-field-diffusion' not in sys.path:
    sys.path.append('/viscam/u/jrshue/neural-field-diffusion')
    
import os
import argparse
import torch
import numpy as np
from dataset import OccupancyDataset, SingleExampleOccupancyDataset, GeneralizeDataset
from torch.utils.data import DataLoader
from networks import CartesianPlaneNonSirenEmbeddingNetwork, TriplaneAutoDecoder
from triplane_diffusion.utils.visualization import save_cross_section, save_shape
from triplane_diffusion.utils.metrics import compute_iou


def main():
    parser = argparse.ArgumentParser(description='Validate auto-decoder with metrics and visuals.')
    parser.add_argument('--data_dir', type=str,
                    help='directory to pull val data from', required=True)
    parser.add_argument('--batch_size', type=int, default=1, required=False,
                    help='number of scenes per batch')
    parser.add_argument('--points_batch_size', type=int, default=500000, required=False,
                    help='number of points per scene, nss and uniform combined')
    parser.add_argument('--load_ckpt_path', type=str, required=True,
                    help='checkpoint to validate')
    parser.add_argument('--save_dir', type=str, required=True,
                    help='directory to save metrics/visuals to')
    parser.add_argument('--resolution', type=int, default=128, required=False,
                    help='triplane resolution')
    parser.add_argument('--channels', type=int, default=32, required=False,
                    help='triplane depth')
    parser.add_argument('--aggregate_fn', type=str, default='sum', required=False,
                    help='function for aggregating triplane features')
    parser.add_argument('--subset_size', type=int, default=None, required=False,
                    help='size of the dataset subset if we\'re training on a subset')
    parser.add_argument('--training_subset_size', type=int, default=None, required=False,
                    help='size of the dataset subset we trained on, to skip seen examples.')
    parser.add_argument('--no_save_cross_section', default=False, required=False, action='store_true',
                    help='Whether to save cross section for each eval')
    parser.add_argument('--no_save_mrc', default=False, required=False, action='store_true',
                    help='Whether to save mrc for each eval')
    parser.add_argument('--save_ply', default=False, required=False, action='store_true',
                    help='Whether to save ply for each eval')
    parser.add_argument('--use_tanh', default=False, required=False, action='store_true',
                    help='Whether to use tanh to clamp triplanes to [-1, 1].')
    args = parser.parse_args()

    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.save_dir, exist_ok=True)

    # Deterministic dataloader
    if args.training_subset_size is None:
        dataloader = DataLoader(
            OccupancyDataset(dataset_path=args.data_dir, points_batch_size=args.points_batch_size, subset_size=args.subset_size), 
            batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=True, pin_memory=True  # Tune num_workers
        )
    else:
        dataloader = DataLoader(
            GeneralizeDataset(dataset_path=args.data_dir, points_batch_size=args.points_batch_size, training_subset_size=args.training_subset_size, subset_size=args.subset_size), 
            batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=True, pin_memory=True  # Tune num_workers
        )
    # dataloader = DataLoader(
    #     SingleExampleOccupancyDataset(dataset_path=args.data_dir), 
    #     batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True  # Tune num_workers
    # )

    # Triplane auto-decoder
    auto_decoder = TriplaneAutoDecoder(resolution=args.resolution, channels=args.channels, how_many_scenes=len(dataloader), input_dim=3, output_dim=1, aggregate_fn=args.aggregate_fn, use_tanh=args.use_tanh).to(device)
    # auto_decoder = CartesianPlaneNonSirenEmbeddingNetwork(input_dim=3, output_dim=1, aggregate_fn=args.aggregate_fn).to(device)

    checkpoint = torch.load(args.load_ckpt_path)
    auto_decoder.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f'Loaded checkpoint from {args.load_ckpt_path}. Evaluating model at epoch {epoch} with loss {loss}...')

    auto_decoder.eval()

    ious = []
    for (obj_idx, pts_sdf) in dataloader:
    # for (coordinates, gt_occupancies) in dataloader:
        
        coordinates, gt_occupancies = pts_sdf[..., 0:3], pts_sdf[..., -1]
        obj_idx, coordinates = obj_idx.int().to(device), coordinates.float().to(device)
        
        # Get logits from model
        pred_occupancies = auto_decoder(obj_idx, coordinates)

        # Calculate IoU
        occ1 = gt_occupancies.cpu().detach().numpy(), 
        occ2 = (pred_occupancies.cpu().detach().numpy() >= 0.5).astype(float)  # Convert to binary occupancy
        iou = compute_iou(occ1=occ1, occ2=occ2)
        print(f'IoU: {iou}')
        ious.append(iou)

        # Save cross-section and mrc
        if not args.no_save_cross_section:
            print(f'saving cross section...')
            save_cross_section(filename=f'{args.save_dir}/cross_section_{obj_idx.item()}_iou_{iou.item()}', model=auto_decoder, obj_idx=obj_idx)
        save_mrc_filename = None
        if not args.no_save_mrc:
            save_mrc_filename = f'{args.save_dir}/mrc_{obj_idx.item()}_iou_{iou.item()}'
            print(f'saving mrc...')
        save_ply_filename = None
        if args.save_ply:
            save_ply_filename = f'{args.save_dir}/ply_{obj_idx.item()}_iou_{iou.item()}'
            print('saving ply...')
        
        save_shape(model=auto_decoder, mrc_filename=save_mrc_filename, ply_filename=save_ply_filename, obj_idx=obj_idx)
        
        # Save metrics to a file
        mean_iou = np.mean(np.array(ious))

        print(f'MEAN IoU: {mean_iou}')
        with open(f'{args.save_dir}/metrics.txt', 'w') as f:
            f.write(f'IoU: {mean_iou}')




if __name__ == "__main__":
    main()

