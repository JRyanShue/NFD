import argparse
import os

from render_utils import scale_to_unit_sphere

import numpy as np
from tqdm import tqdm
import trimesh


def main(args):

    mesh_files = [
        os.path.join(args.mesh_dir, f) for f in os.listdir(args.mesh_dir)
        if f.endswith(args.suffix)
    ]

    for i, mesh_file in tqdm(enumerate(mesh_files)):
        mesh = trimesh.load_mesh(mesh_file, file_type=args.suffix)
        mesh = scale_to_unit_sphere(mesh)
        pc = trimesh.sample.sample_surface(mesh, count=args.num_points)
        pc = np.array(pc[0])

        np.save(os.path.join(args.pc_dir, f"pc-{i}.npy"), pc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-dir", type=str, required=True)
    parser.add_argument("--pc-dir", type=str, required=True)
    parser.add_argument("--suffix", type=str, default="off")
    parser.add_argument("--num-points", type=int, default=2048)
    args = parser.parse_args()

    os.makedirs(args.pc_dir, exist_ok=True)
    main(args)
