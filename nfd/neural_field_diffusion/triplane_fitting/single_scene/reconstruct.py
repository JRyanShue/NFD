
'''
Reconstruct a single scene from a pretrained triplane + decoder.

'''

import sys
if '/viscam/u/jrshue/neural-field-diffusion' not in sys.path:
    sys.path.append('/viscam/u/jrshue/neural-field-diffusion')

import os
import numpy as np
import argparse
import torch
from triplane_diffusion.networks import CartesianPlaneNonSirenEmbeddingNetwork
from triplane_diffusion.utils.visualization import save_cross_section, save_mrc


def main():
    parser = argparse.ArgumentParser(description='Sample from given checkpoint.')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default=1, required=True,
                    help='directory to save metrics/visuals to')
    parser.add_argument('--aggregate_fn', type=str, required=True,
                    help='function for aggregating triplane features')
    parser.add_argument('--data_dir', type=str, default=None, required=False, 
                    help='where to pull val data from to generate metrics. If not specified, will not generate metrics.')
    parser.add_argument('--batch_size', type=int, default=1, required=False,
                    help='number of scenes per batch')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    # Load in auto-decoder with triplane
    auto_decoder = CartesianPlaneNonSirenEmbeddingNetwork(input_dim=3, output_dim=1, aggregate_fn=args.aggregate_fn).to(device)
    checkpoint = torch.load(args.checkpoint_path)
    auto_decoder.load_state_dict(checkpoint['model_state_dict'])
    auto_decoder.eval()

    # Metrics: IoU, Chamfer Distance, F-score
    if args.data_dir:
        # Import dataset for loading val data
        from triplane_diffusion.dataset import OccupancyDataset, SingleExampleOccupancyDataset
        from torch.utils.data import DataLoader
        from triplane_diffusion.utils.metrics import compute_singleshape_mean_iou

        # List of ious to collapse into mean later
        ious = []
        # For single-shape expt.
        dataloader = DataLoader(
            SingleExampleOccupancyDataset(dataset_path=args.data_dir, validation=True), 
            batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True  # Tune num_workers
        )
        mean_iou = compute_singleshape_mean_iou(model=auto_decoder, dataloader=dataloader)
        
        print(f'MEAN IoU: {mean_iou}')
        with open(f'{args.save_dir}/metrics.txt', 'w') as f:
            f.write(f'IoU: {mean_iou}')

    # Save cross-section and mrc
    print(f'saving cross section...')
    save_cross_section(f'{args.save_dir}/cross_section', auto_decoder)
    print(f'saving mrc...')
    save_mrc(f'{args.save_dir}/mrc', auto_decoder)



if __name__ == "__main__":
    main()

