


'''
Loop through dataset to test I/O properties.

'''

import sys
if '/viscam/u/jrshue/neural-field-diffusion' not in sys.path:
    sys.path.append('/viscam/u/jrshue/neural-field-diffusion')
    
import argparse
import torch
from triplane_diffusion.dataset import OccupancyDataset
from torch.utils.data import DataLoader
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Loop through dataset to test I/O properties.')
    parser.add_argument('--data_dir', type=str,
                    help='directory to pull data from', required=True)
    parser.add_argument('--batch_size', type=int, default=1, required=False,
                    help='number of scenes per batch')
    parser.add_argument('--points_batch_size', type=int, default=500000, required=False,
                    help='number of points per scene, nss and uniform combined')
    parser.add_argument('--subset_size', type=int, default=None, required=False,
                    help='size of the dataset subset if we\'re training on a subset')
    args = parser.parse_args()

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    # Deterministic dataloader
    dataloader = DataLoader(
        OccupancyDataset(dataset_path=args.data_dir, points_batch_size=args.points_batch_size, subset_size=args.subset_size), 
        batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True, pin_memory=True  # Tune num_workers
    )

    N_EPOCHS = 1
    step = 0
    load_start_time = time.time()

    time_to_loads = []
    time_to_transfers = []

    for epoch in range(N_EPOCHS):
        print(f'EPOCH {epoch}...')
        for (obj_idx, pts_sdf) in dataloader:
        # for (coordinates, gt_occupancies) in dataloader:

            time_to_load = time.time() - load_start_time
            print(f'Time to load data with CPU for obj_idx {obj_idx}: {time_to_load}')
            time_to_loads.append(time_to_load)

            start_time = time.time()

            obj_idx, pts_sdf = obj_idx.int().to(device), pts_sdf.float().to(device)

            time_to_transfer = time.time() - start_time
            print(f'Time to move data from CPU to GPU: {time_to_transfer}')
            time_to_transfers.append(time_to_transfer)

            # # Sample on GPU
            # sample_indices = torch.Tensor(args.batch_size, args.points_batch_size).uniform_(0, pts_sdf.shape[1]).long().to(device)  # .expand_as(pts_sdf)
            # # TODO @JRyanShue or @zankner: use torch.gather() or a better function for this to get rid of for loop. Though it only loops over batch_size so it's not too bad.
            # sampled_pts_sdf = torch.cat([batch_elem[index_elem].unsqueeze(0) for batch_elem, index_elem in zip(pts_sdf, sample_indices)])
            
            # coordinates, gt_occupancies = sampled_pts_sdf[..., 0:3], sampled_pts_sdf[..., -1]
            coordinates, gt_occupancies = pts_sdf[..., 0:3], pts_sdf[..., -1]
            # print(f'coordinates.shape: {coordinates.shape}, gt_occupancies.shape: {gt_occupancies.shape}')

            load_start_time = time.time()
    
    print(f'Average time to load a batch of size {args.batch_size} with point batch size {args.points_batch_size} into CPU: {np.mean(np.array(time_to_loads))}')
    print(f'Average time to transfer a batch of size {args.batch_size} with point batch size {args.points_batch_size} from CPU to GPU: {np.mean(np.array(time_to_transfers))}')




if __name__ == "__main__":
    main()

