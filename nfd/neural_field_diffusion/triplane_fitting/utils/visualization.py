
import mrcfile
import numpy as np
import torch
import matplotlib.pyplot as plt


device = ('cuda' if torch.cuda.is_available() else 'cpu')


def save_cross_section(filename, model, obj_idx=None, res=512, max_batch_size=50000):
    # Output a cross section of the fitted volume

    xx = torch.linspace(-1, 1, res)
    yy = torch.linspace(-1, 1, res)
    # zz = torch.linspace(-1, 1, res)
    (x_coords, y_coords) = torch.meshgrid([xx, yy])
    z_coords = torch.zeros_like(x_coords)
    coords = torch.cat([x_coords.unsqueeze(-1), y_coords.unsqueeze(-1), z_coords.unsqueeze(-1)], -1)

    coords = coords.reshape(res*res, 3)
    prediction = torch.zeros(coords.shape[0], 1)

    with torch.no_grad():
        head = 0
        while head < coords.shape[0]:
            input_params = {}
            if obj_idx is not None:
                input_params['obj_idx'] = obj_idx
            input_params['coordinates'] = coords[head:head+max_batch_size].to(device).unsqueeze(0)

            prediction[head:head+max_batch_size] = model(**input_params).cpu()
            head += max_batch_size
            
    prediction = (prediction > 0).cpu().numpy().astype(np.uint8)
    prediction = prediction.reshape(res, res)
    plt.imshow(prediction)
    plt.savefig(f'{filename}.png')

# cross_section(model, 512)


def save_shape(model, mrc_filename=None, ply_filename=None, obj_idx=None, res=256, max_batch_size=50000, mrc_mode=2):
    # Output a res x res x res x 1 volume prediction. Download ChimeraX to open the files.
    # Set the threshold in ChimeraX to 0.5 if mrc_mode=0, 0 else

    model.eval()
    xx = torch.linspace(-1, 1, res)
    yy = torch.linspace(-1, 1, res)
    zz = torch.linspace(-1, 1, res)
    (x_coords, y_coords, z_coords) = torch.meshgrid([xx, yy, zz])
    coords = torch.cat([x_coords.unsqueeze(-1), y_coords.unsqueeze(-1), z_coords.unsqueeze(-1)], -1)

    coords = coords.reshape(res*res*res, 3)
    prediction = torch.zeros(coords.shape[0], 1)
    
    with torch.no_grad():
        head = 0
        while head < coords.shape[0]:
            input_params = {}
            if obj_idx is not None:
                input_params['obj_idx'] = obj_idx
            input_params['coordinates'] = coords[head:head+max_batch_size].to(device).unsqueeze(0)

            prediction[head:head+max_batch_size] = model(**input_params).cpu()
            head += max_batch_size
            
    if mrc_mode == 0:
        prediction = (prediction > 0).cpu().numpy().astype(np.uint8)
    print("Writing Data")


    # if mrc_output_path.split('.')[-1] == 'gz':
    #     with mrcfile.new(f'{filename}.mrc', overwrite=True, compression='gzip') as mrc:
    #         mrc.set_data(prediction.reshape(res, res, res).numpy())
    # else:
    if mrc_filename is not None:
        with mrcfile.new_mmap(f'{mrc_filename}.mrc', overwrite=True, shape=(res, res, res), mrc_mode=mrc_mode) as mrc:
            mrc.data[:] = prediction.reshape(res, res, res)
    if ply_filename is not None:
        from utils.shape_utils import convert_sdf_samples_to_ply
        convert_sdf_samples_to_ply(np.transpose(prediction.reshape(res, res, res).cpu().numpy(), (2, 1, 0)), [0, 0, 0], 1, f'{ply_filename}.ply', level=0.5)