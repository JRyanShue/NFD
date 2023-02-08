import torch
import numpy as np
import blobfile as bf


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

class OccupancyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        self.data = np.load(dataset_path)
        self.data = torch.tensor(self.data.reshape(50, -1, 4)).cuda()
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx,:, :3], self.data[idx, :, 3:]
    
    
    
class MultiOccupancyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, subset_size = 100, device = "cuda"):
        self.file_list = _list_image_files_recursively(dataset_path)[:subset_size]
        self.subset_size = subset_size
        self.device = device
        
        data_list = []
        for file_path in self.file_list:
            curr_data = np.load(file_path)
            curr_data = torch.Tensor(curr_data)
            data_list.append(curr_data.reshape(-1, *curr_data.shape))
            
        self.data = torch.Tensor(torch.cat(data_list)).to(self.device)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return idx, self.data[idx,:, :3], self.data[idx, :, 3:]