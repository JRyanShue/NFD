import argparse
import trimesh
import numpy as np
from inside_mesh import inside_mesh
from pykdtree.kdtree import KDTree
from tqdm import tqdm
import torch
#import mcubes

def normalize_mesh(mesh):
    print("Scaling Parameters: ", mesh.bounding_box.extents)
    mesh.vertices -= mesh.bounding_box.centroid
    mesh.vertices /= np.max(mesh.bounding_box.extents / 2)

    
def compute_volume_points(intersector, count, max_batch_size = 1000000):
    coordinates = np.random.rand(count, 3) * 2 - 1
    
    occupancies = np.zeros((count, 1), dtype=int)
    head = 0
    with tqdm(total = coordinates.shape[0]) as pbar:
        while head < coordinates.shape[0]:
            occupancies[head:head+max_batch_size] = intersector.query(coordinates[head:head+max_batch_size]).astype(int).reshape(-1, 1)
            head += max_batch_size
            pbar.update(min(max_batch_size, coordinates.shape[0] - head))
        return np.concatenate([coordinates, occupancies], -1)
    
def compute_near_surface_points(mesh, intersector, count, epsilon, max_batch_size = 1000000):
    coordinates = trimesh.sample.sample_surface(mesh, count)[0] + np.random.randn(*(count, 3)) * epsilon

    occupancies = np.zeros((count, 1), dtype=int)
    head = 0
    with tqdm(total = coordinates.shape[0]) as pbar:
        while head < coordinates.shape[0]:
            occupancies[head:head+max_batch_size] = intersector.query(coordinates[head:head+max_batch_size]).astype(int).reshape(-1, 1)
            head += max_batch_size
            pbar.update(min(max_batch_size, coordinates.shape[0] - head))
    return np.concatenate([coordinates, occupancies], -1)

def compute_obj(mesh, intersector, max_batch_size = 1000000, res = 1024):
    xx = torch.linspace(-1, 1, res)
    yy = torch.linspace(-1, 1, res)
    zz = torch.linspace(-1, 1, res)

    (x_coords, y_coords, z_coords) = torch.meshgrid([xx, yy, zz])
    coords = torch.cat([x_coords.unsqueeze(-1), y_coords.unsqueeze(-1), z_coords.unsqueeze(-1)], -1)

    coordinates = coords.reshape(res*res*res, 3).numpy()

    occupancies = np.zeros((res*res*res, 1), dtype=int)
    head = 0
    with tqdm(total = coordinates.shape[0]) as pbar:
        while head < coordinates.shape[0]:
            occupancies[head:head+max_batch_size] = intersector.query(coordinates[head:head+max_batch_size]).astype(int).reshape(-1, 1)
            head += max_batch_size
            pbar.update(min(max_batch_size, coordinates.shape[0] - head))
    
    occupancies = occupancies.reshape(res, res, res)
    vertices, triangles = mcubes.marching_cubes(occupancies, 0)
    mcubes.export_obj(vertices, triangles, "data/car_gt_"+str(res)+".obj")


def generate_gt_obj(filepath):
    print("Loading mesh...")
    mesh = trimesh.load(filepath, process=False, force='mesh', skip_materials=True)
    normalize_mesh(mesh)
    intersector = inside_mesh.MeshIntersector(mesh, 2048)

    compute_obj(mesh, intersector, res = 1024)

def generate_volume_dataset(filepath, output_filepath, num_surface, epsilon):
    print("Loading mesh...")
    mesh = trimesh.load(filepath, process=False, force='mesh', skip_materials=True)
    normalize_mesh(mesh)
    intersector = inside_mesh.MeshIntersector(mesh, 2048)

    print("Computing near surface points...")
    surface_points = compute_near_surface_points(mesh, intersector, num_surface, epsilon)

    print("Computing volume points...")
    volume_points = compute_volume_points(intersector, num_surface)

    all_points = np.concatenate([surface_points, volume_points], 0)
    np.save(output_filepath, all_points)

def generate_border_occupancy_dataset(filepath, output_filepath, count, wall_thickness = 0.0025):
    mesh = trimesh.load(filepath, process=False, force='mesh', skip_materials=True)
    normalize_mesh(mesh)
    
    surface_points, _ = trimesh.sample.sample_surface(mesh, 10000000)
    kd_tree = KDTree(surface_points)
    
    volume_points = np.random.randn(count, 3) * 2 - 1
    dist, _ = kd_tree.query(volume_points, k=1)
    volume_occupancy = np.where(dist < wall_thickness, np.ones_like(dist), np.zeros_like(dist))
    
    near_surface_points = trimesh.sample.sample_surface(mesh, count)[0] + np.random.randn(count, 3) * EPSILON
    dist, _ = kd_tree.query(near_surface_points, k=1)
    near_surface_occupancy = np.where(dist < wall_thickness, np.ones_like(dist), np.zeros_like(dist))
    
    points = np.concatenate([volume_points, near_surface_points], 0)
    occ = np.concatenate([volume_occupancy, near_surface_occupancy], 0).reshape(-1, 1)
    
    dataset = np.concatenate([points, occ], -1)
    np.save(output_filepath, dataset)
    
    
    
# NUMSURFACE=20000000
# NUMVOLUME= 20000000
EPSILON=0.01

# FILEPATH = '/media/data5/lindell/castle.obj'
# FILEPATH = 'meshes/thai_statue.ply'
# OUTPUTFILEPATH = 'meshes/castle_surface_20m.npy'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--type', type=str, choices = ['volume', 'border', 'gt'], default='volume')
    parser.add_argument('--count', type=int, default=20000000)
    
    args = parser.parse_args()
    
    if args.type == 'volume':
        generate_volume_dataset(args.input, args.output, args.count, EPSILON)
    elif args.type == 'border':
        generate_border_occupancy_dataset(args.input, args.output, args.count)
    elif args.type == "gt":
        generate_gt_obj(args.input)