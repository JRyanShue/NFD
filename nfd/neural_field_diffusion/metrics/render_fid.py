import argparse
import os

from render_utils import Render, create_pose, scale_to_unit_sphere

import numpy as np
from tqdm import tqdm
import torch
import torchvision
import trimesh


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


FrontVector = (np.array(
    [[0.52573, 0.38197, 0.85065], [-0.20081, 0.61803, 0.85065],
     [-0.64984, 0.00000, 0.85065], [-0.20081, -0.61803, 0.85065],
     [0.52573, -0.38197, 0.85065], [0.85065, -0.61803, 0.20081],
     [1.0515, 0.00000, -0.20081], [0.85065, 0.61803, 0.20081],
     [0.32492, 1.00000, -0.20081], [-0.32492, 1.00000, 0.20081],
     [-0.85065, 0.61803, -0.20081], [-1.0515, 0.00000, 0.20081],
     [-0.85065, -0.61803, -0.20081], [-0.32492, -1.00000, 0.20081],
     [0.32492, -1.00000, -0.20081], [0.64984, 0.00000, -0.85065],
     [0.20081, 0.61803, -0.85065], [-0.52573, 0.38197, -0.85065],
     [-0.52573, -0.38197, -0.85065], [0.20081, -0.61803, -0.85065]])) * 2


def render_mesh(mesh,
                resolution=1024,
                index=5,
                background=None,
                scale=1,
                no_fix_normal=True):

    camera_pose = create_pose(FrontVector[index] * scale)

    render = Render(size=resolution,
                    camera_pose=camera_pose,
                    background=background)

    triangle_id, rendered_image, normal_map, depth_image, p_images = render.render(
        path=None, clean=True, mesh=mesh, only_render_images=no_fix_normal)
    return rendered_image


def render_for_fid(mesh, root_dir, mesh_idx):
    render_resolution = 299
    mesh = scale_to_unit_sphere(mesh)
    for j in range(20):
        image = render_mesh(mesh, index=j, resolution=render_resolution) / 255
        torchvision.utils.save_image(
            torch.from_numpy(image.copy()).permute(2, 0, 1),
            f"{root_dir}/view_{j}/{mesh_idx}.png")


def main(args):
    gen_meshes = get_meshes(args.gen_dir)
    gt_meshes = get_meshes(args.gt_dir)

    for i in range(20):
        os.mkdir(os.path.join(args.gt_out_dir, f"view_{i}"))
        os.mkdir(os.path.join(args.gen_out_dir, f"view_{i}"))

    for mesh_idx, gen_mesh in tqdm(enumerate(gen_meshes),
                                   total=len(gen_meshes)):
        render_for_fid(gen_mesh, args.gen_out_dir, mesh_idx)

    for mesh_idx, gt_mesh in tqdm(enumerate(gt_meshes), total=len(gt_meshes)):
        render_for_fid(gt_mesh, args.gt_out_dir, mesh_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", type=str, required=True)
    parser.add_argument("--gen-dir", type=str, required=True)
    parser.add_argument("--gt-out-dir", type=str, required=True)
    parser.add_argument("--gen-out-dir", type=str, required=True)
    args = parser.parse_args()

    main(args)
