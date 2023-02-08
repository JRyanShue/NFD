import argparse
import os

from evaluation_metrics import compute_all_mesh_metrics

import numpy as np
from tqdm import tqdm
import torch
import trimesh

# def get_meshes(data_dir):
#     pc_files = [
#         os.path.join(data_dir, f) for f in os.listdir(data_dir)
#         if f.endswith("npy")
#     ]

#     pcs = []
#     for pc_file in tqdm(pc_files):
#         pc = np.load(pc_file)
#         pcs.append(pc)

#     return torch.from_numpy(
#         np.vstack(pcs).reshape(-1, pc.shape[0],
#                                pc.shape[1])).type(torch.float32)


def get_meshes(data_dir):
    mesh_files = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.endswith("obj") or f.endswith("off")
    ]
    file_suffix = mesh_files[0].split(".")[-1]

    meshes = [
        trimesh.load_mesh(mesh_file, file_type=file_suffix)
        for mesh_file in mesh_files
    ]

    return meshes


def main(args):
    gen_pcs = get_meshes(args.gen_dir)
    gt_pcs = get_meshes(args.gt_dir)

    results = compute_all_mesh_metrics(gen_pcs, gt_pcs, args.batch_size)
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", type=str, required=True)
    parser.add_argument("--gen-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    main(args)
