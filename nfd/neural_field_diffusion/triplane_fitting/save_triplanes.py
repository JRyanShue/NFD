


'''
Take a model checkpoint and save out its embedding matrix as a dataset of triplanes.

'''

import sys
if '/viscam/u/jrshue/neural-field-diffusion' not in sys.path:
    sys.path.append('/viscam/u/jrshue/neural-field-diffusion')
    
import os
import argparse
import numpy as np
import torch
from networks import TriplaneAutoDecoder
import time



def main():
    parser = argparse.ArgumentParser(description='Take a model checkpoint and save out its embedding matrix as a dataset of triplanes.')
    parser.add_argument('--load_ckpt_path', type=str, required=True,
                    help='checkpoint to pull model from')
    parser.add_argument('--outdir', type=str, required=True,
                    help='where to save model triplanes')
    parser.add_argument('--resolution', type=int, default=128, required=False,
                    help='triplane resolution')
    parser.add_argument('--channels', type=int, default=32, required=False,
                    help='triplane depth')
    parser.add_argument('--aggregate_fn', type=str, default='sum', required=False,
                    help='function for aggregating triplane features')
    parser.add_argument('--subset_size', type=int, required=True,
                    help='size of the dataset subset we trained on')
    parser.add_argument('--subset_start_idx', type=int, default=0, required=False,
                    help='if we took a shifted subset, where in the dataset we start. For saving triplanes with the right index.')
    parser.add_argument('--use_tanh', default=False, required=False, action='store_true',
                    help='Whether to use tanh to clamp triplanes to [-1, 1].')
    args = parser.parse_args()

    # device = ('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    os.makedirs(args.outdir, exist_ok=True)

    # Triplane auto-decoder
    auto_decoder = TriplaneAutoDecoder(resolution=args.resolution, channels=args.channels, how_many_scenes=args.subset_size, input_dim=3, output_dim=1, aggregate_fn=args.aggregate_fn).to(device)
    
    
    if args.load_ckpt_path:
        checkpoint = torch.load(args.load_ckpt_path)
        auto_decoder.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded checkpoint from {args.load_ckpt_path}. Extracting triplanes...')

    auto_decoder.train()

    # Save triplanes to outdir
    for idx, triplane in enumerate(auto_decoder.embeddings.weight):
        print(f'Saving to {args.outdir}/{str(idx + args.subset_start_idx).zfill(4)}.npy...')
        np.save(f'{args.outdir}/{str(idx + args.subset_start_idx).zfill(4)}', triplane.detach().numpy().reshape(3, args.channels, args.resolution, args.resolution))


if __name__ == "__main__":
    main()

