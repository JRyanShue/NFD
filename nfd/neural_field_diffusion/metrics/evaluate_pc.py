import argparse
import os

from evaluation_metrics import compute_all_pc_metrics

import numpy as np
from tqdm import tqdm
import torch

def get_pcs(data_dir):
    pc_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith("npy")]

    pcs = []
    for pc_file in tqdm(pc_files):
        pc = np.load(pc_file)
        pcs.append(pc)
    
    return torch.from_numpy(np.vstack(pcs).reshape(-1, pc.shape[0], pc.shape[1])).type(torch.float32)

def main(args):
    gen_pcs = get_pcs(args.gen_dir)
    gt_pcs = get_pcs(args.gt_dir)
    
    results = compute_all_pc_metrics(gen_pcs, gt_pcs, args.batch_size)
    print(results)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", type=str, required=True)
    parser.add_argument("--gen-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    main(args)
