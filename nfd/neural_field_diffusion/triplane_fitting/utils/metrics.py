
import numpy as np
import torch
# from imt.util.libkdtree import KDTree

# Adapted from Occupancy Networks code
def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # print(f'occ1[0]: {occ1[0]}')

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou


def compute_singleshape_mean_iou(model, dataloader, device=('cuda' if torch.cuda.is_available() else 'cpu')):
    ious = []
    for (coordinates, gt_occupancies) in dataloader:

        coordinates, gt_occupancies = coordinates.float().to(device), gt_occupancies.float().to(device)
        pred_occupancies = model(coordinates)

        occ1 = gt_occupancies.cpu().detach().numpy(), 
        occ2 = (pred_occupancies.cpu().detach().numpy() >= 0.5).astype(float)  # Convert to binary occupancy
        iou = compute_iou(occ1=occ1, occ2=occ2)
        print(f'IoU: {iou}')
        ious.append(iou)
    
    return np.mean(np.array(ious))
