

import torch
import numpy as np
import blobfile as bf
import time
import json
import os

import importlib
wisp_multiview_dataset = importlib.import_module('modules.kaolin-wisp.wisp.datasets.multiview_dataset')
wisp_core = importlib.import_module('modules.kaolin-wisp.wisp.core')

device = ('cuda' if torch.cuda.is_available() else 'cpu')


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "npy"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


# Occupancy trains the model better than SDFs
def sdf_to_occ(arr):
    return (arr > 0.5).astype(float)



class BigTensorOccupancyDataset():
    def __init__(self, dataset_path, batch_size, points_batch_size, subset_size=None, validation=False, device='cuda'):
        self.device = device
        self.validation = validation
        self.filenames = _list_image_files_recursively(dataset_path)
        self.subset_size = subset_size
        if self.subset_size is not None:
            self.filenames = self.filenames[:self.subset_size]
        self.batch_size = batch_size
        self.points_batch_size = points_batch_size

        # Retrieve data from disk by indexing into the filenames. This way we can operate in tensor space.
        print(f'Loading {len(self.filenames)} files into a big tensor...')
        self.data = self.load_data()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, obj_idx):
        return self.data[obj_idx]

    def load_data(self):
        tensor_list = []
        
        for obj_filepath in self.filenames:

            obj_data = np.load(obj_filepath)
            # Only use uniform samples if using for validation
            if self.validation: 
                obj_data = obj_data[int(obj_data.shape[0]/2):]
            if self.points_batch_size != obj_data.shape[0]:
                print('Sampling...')  # We usually don't want to have to sample
                indices = np.random.randint(low=0, high=obj_data.shape[0], size=self.points_batch_size)
                sampled_data = obj_data[indices] # self.points_batch_size, 4
            else:
                sampled_data = obj_data

            data_tensor = torch.Tensor(sampled_data).to(self.device)
            data_tensor = data_tensor.reshape(-1, *data_tensor.shape)
            tensor_list.append(data_tensor)
        
        return torch.Tensor(torch.cat(tensor_list))


class OccupancyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, points_batch_size, subset_size=None, validation=False):
        self.data = _list_image_files_recursively(dataset_path)  # List of all filenames to include
        self.subset_size = subset_size
        if self.subset_size is not None:
            self.data = self.data[:self.subset_size]
        self.points_batch_size = points_batch_size
        self.validation = validation
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, obj_idx):
        # return self.data[idx,:, :3], self.data[idx, :, 3:]

        # load the object npz (or mesh) 
        start_time = time.time()
        obj_filepath = self.data[obj_idx]

        obj_data = np.load(obj_filepath) # e.g. 1000000, 4 -> xyz occupancy
        # print(f'Load time: {time.time() - start_time}')
        # Only use uniform samples if using for validation
        if self.validation: 
            obj_data = obj_data[int(obj_data.shape[0]/2):]

        # Sample points -- slow because on CPU
        start_time = time.time()
        if self.points_batch_size != obj_data.shape[0]:
            indices = np.random.randint(low=0, high=obj_data.shape[0], size=self.points_batch_size)
            print(f'Getting indices time: {time.time() - start_time}')
            start_time = time.time()
            sampled_data = obj_data[indices] # self.points_batch_size, 4
        else:
            sampled_data = obj_data
        # print(f'Sampling time: {time.time() - start_time}')
        # start_time = time.time()
        # coordinates = sampled_data[:, 0:3]
        # occupancies = sampled_data[:, -1]
        # print(f'Splitting time: {time.time() - start_time}')

        # start_time = time.time()
        # coordinates = obj_data[:, 0:3]
        # occupancies = obj_data[:, -1]
        # print(f'Splitting time: {time.time() - start_time}')

        # return obj_idx, coordinates, occupancies # (10M, 3), (10M, 1)  # (100000, 3), (100000, 1)
        # return obj_idx, obj_data  # (10M, 4)
        return obj_idx, sampled_data  # ex: (100-500K, 4)


'''
Dataset will skip over files seen at training time.
'''
class GeneralizeDataset(OccupancyDataset):
    def __init__(self, dataset_path, points_batch_size, training_subset_size=500, subset_size=1, validation=False):
        super().__init__(dataset_path, points_batch_size, subset_size, validation=False)
        self.data = _list_image_files_recursively(dataset_path)
        self.training_subset_size = training_subset_size
        if self.training_subset_size is not None:
            self.data = self.data[self.training_subset_size:]
        if self.subset_size is not None:
            self.data = self.data[:self.subset_size]


# For overfitting on a single scene
class SingleExampleOccupancyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, validation=False):
        self.data = np.load(dataset_path)
        if validation:
            self.data = self.data[int(self.data.shape[0]/2):]

        self.points_per_slice = self.data.shape[0] / 100
        print(f'Points per slice: {self.points_per_slice}')

        np.random.shuffle(self.data)
        self.data = self.data.reshape(500, -1, 4)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx,:, :3], self.data[idx, :, 3:]  # sdf_to_occ(self.data[idx, :, 3:])


'''
Making this as simple as possible.
Take in a path to a json file in the EG3D format. Return images along with their attributes.
Subclass of kaolin-wisp MultiviewDataset, with changes to work for multiple scenes

args:
    dataset_path: path to single-scene dataset in the RTMV data format

'''
class SingleSceneNerfDataset(wisp_multiview_dataset.MultiviewDataset):
    def __init__(self, dataset_path):
        super().__init__(dataset_path=dataset_path, multiview_dataset_format='rtmv')


'''
For now, this class will load all data into memory to minimize I/O bottlenecks.

TODO (JRyanShue): make a function for loading the data onto GPU memory for even less bottlenecking.

args:
    dataset_path: path to directory of single-scene datasets in the RTMV data format

'''
class NerfDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, subset_size, ray_batch_size=4000, ds_type='train', sample_points=True):
        self.root = dataset_path
        self.subset_size = subset_size
        self.ray_batch_size = ray_batch_size
        # self.ds_type = ds_type
        self.sample_points = sample_points  # Whether to sample. If False, will just return the entire image for as many images we can fit into memory.
        self.scene_dirs = [f'{self.root}/{scene_idx}' for scene_idx in sorted(os.listdir(dataset_path))][:self.subset_size]
        self.data = [SingleSceneNerfDataset(dataset_path=scene_dir) for scene_dir in self.scene_dirs]
        self.val_data = []
        self.data_split = 'train'

    def __len__(self):
        return len(self.data)

    def init(self):
        for scene in self.data:
            # Initialize train data
            scene.init()

            # Initialize val data
            scene_val_imgs = scene.get_images(split='val', mip=0)
            # Flatten into an array of rays/image GT values
            num_imgs = scene_val_imgs['rays'].shape[0]
            scene_val_imgs['rays'] = scene_val_imgs['rays'].reshape(num_imgs, -1, 3)
            scene_val_imgs['imgs'] = scene_val_imgs['imgs'].reshape(num_imgs, -1, 3)
            self.val_data.append(scene_val_imgs)

    def train(self):
        self.data_split = 'train'

    def val(self):
        self.data_split = 'val'

    def __getitem__(self, idx):
        obj_idx = idx
        obj_idx = torch.tensor([idx]).to(device)
        obj_idx = obj_idx.reshape(-1, obj_idx.shape[0])
        data = self.data[idx].data

        if self.data_split == 'train':
            data = self.data[idx].data
        elif self.data_split == 'val':
            data = self.val_data[idx]

        if self.sample_points:
            # Sample data. Equal number of rays from each image.
            ray_idxs = torch.randint(low=0, high=data['rays'].shape[-1], size=(data['rays'].shape[0], self.ray_batch_size // data['rays'].shape[0]))  # e.g. torch.Size([66, 60])
            selected_img_gts = torch.stack([img[ray_idx] for ray_idx, img in zip(ray_idxs, data['imgs'])])  # e.g. [66, 60, 3]
            selected_rays = wisp_core.Rays.stack([img_rays[ray_idx] for ray_idx, img_rays in zip(ray_idxs, data['rays'])])  # e.g. [66, 60]
            
            data['imgs'] = selected_img_gts
            data['rays'] = selected_rays
        else:
            # Return one to a few whole images
            data['imgs'] = data['imgs']  # [:3]
            data['rays'] = data['rays']  # [:3]

        return obj_idx, data

        # TODO(JRyanShue): split this into two datasets, so the data-heavy one can be used with a dataloader and the other can use kaolin-wisp's conventions

        # print(len(list(data['cameras'].values())))  # 66
        # print(list(data['cameras'].values())[0].__dict__)
        # print(list(data['cameras'].values())[0].intrinsics.__dict__, '\n\n', list(data['cameras'].values())[0].extrinsics.__dict__)
        # return dict(
        #     imgs=data['imgs'],  # e.g. [66, 16384, 3] ([66, 128^2, 3])
        #     masks=data['masks'],  # e.g. [66, 16384, 1]
        #     rays=dict(origins=data['rays'].origins,  # e.g. [66, 16384, 3]
        #     dirs=data['rays'].dirs),  # e.g. [66, 16384, 3]
        #     depths=data['depths'],  # e.g. [66, 16384, 1]
        #     cameras=list(data['cameras'].values())
        # )