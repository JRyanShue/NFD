

import torch
import torch.nn as nn
import numpy as np

# from triplane_fitting.utils.volumetric_rendering.renderer import ImportanceRenderer
# from triplane_fitting.utils.volumetric_rendering.ray_sampler import RaySampler

import importlib
wisp_embedders = importlib.import_module('modules.kaolin-wisp.wisp.models.embedders')


def first_layer_sine_init(m):
    with torch.no_grad():
        # if hasattr(m, 'weight'):
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)

def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            # if hasattr(m, 'weight'):
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
    return init




class TriplaneAutoDecoder(nn.Module):
    def __init__(
        self, 
        resolution, 
        channels, 
        how_many_scenes, 
        input_dim=3, 
        output_dim=1, 
        aggregate_fn='prod',  # vs sum
        use_tanh=False, 
        view_embedding=True,
        rendering_kwargs={},
        neural_rendering_resolution=64,
        triplane_cpu_intermediate=False,
        device='cpu',
        
    ):
        super().__init__()

        self.aggregate_fn = aggregate_fn
        print(f'Using aggregate_fn {aggregate_fn}')

        self.resolution = resolution
        self.channels = channels
        self.embeddings = [torch.nn.Embedding(1, 3 * self.channels * self.resolution * self.resolution) for i in range(how_many_scenes)]  # , sparse=True)
        self.use_tanh = use_tanh
        self.view_embedding = view_embedding
        self.embedder_type = 'positional'
        self.view_multires = 4
        self.triplane_cpu_intermediate = triplane_cpu_intermediate
        self.device = device

        if not self.triplane_cpu_intermediate:
            for embedding in self.embeddings:
                embedding = embedding.to(self.device)

        if output_dim == 1:
            self.mode = 'shape'
        elif output_dim == 4:
            self.mode = 'nerf'
        else: 
            raise Exception('Invalid output dimensionality.')

        if self.mode == 'nerf':
            if self.view_embedding:
                print('Initializing view embedding...')
                self.view_embedder, self.view_embed_dim = wisp_embedders.get_positional_embedder(self.view_multires, 
                                                                         self.embedder_type == 'positional')
            else:
                self.view_embed_dim = 0
            # self.neural_rendering_resolution = neural_rendering_resolution
            # self.renderer = ImportanceRenderer()
            # self.ray_sampler = RaySampler()

        # Lightweight decoder
        self.net = nn.Sequential(
            # https://arxiv.org/abs/2006.10739 - Fourier FN

            nn.Linear(self.channels + self.view_embed_dim, 128),
            nn.ReLU(),
            
            nn.Linear(128, 128),
            nn.ReLU(),
            
            nn.Linear(128, output_dim),
        ).to(self.device)
        
        self.net.apply(frequency_init(30))
        self.net[0].apply(first_layer_sine_init)

        self.rendering_kwargs = rendering_kwargs

        if self.triplane_cpu_intermediate:
            # We need to store the currently used triplanes on GPU memory, but don't want to load them each time we make a forward pass.
            self.current_embeddings = None  # Embedding object within list of embeddings. Need this intermediate step for gradient to pass through
            self.current_triplanes = None
            self.current_obj_idx = None


    def sample_plane(self, coords2d, plane):
        assert len(coords2d.shape) == 3, coords2d.shape
        sampled_features = torch.nn.functional.grid_sample(plane,
                                                           coords2d.reshape(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
                                                           mode='bilinear', padding_mode='zeros', align_corners=True)
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.reshape(N, C, H*W).permute(0, 2, 1)
        return sampled_features


    def forward(self, obj_idx, coordinates, ray_d=None, debug=False):

        # print(f'coordinates.shape: {coordinates.shape}')  # e.g. [1, 16359, 256, 3]
        
        if len(coordinates.shape) == 3:
            batch_size, n_coords, n_dims = coordinates.shape
        elif len(coordinates.shape) == 4:
            batch_size, ray_batch_size, n_coords, n_dims = coordinates.shape
        assert batch_size == obj_idx.shape[0]

        # Get embedding at index and reshape to (N, 3, channels, H, W)
        # self.embeddings[obj_idx].to(self.device)
        # print(f'obj_idx: {obj_idx}')  # e.g. tensor([[0]], device='cuda:0')

        if self.triplane_cpu_intermediate:
            # Move triplane from CPU to GPU. Only happens once per scene.
            if obj_idx != self.current_obj_idx:
                print(f'Moving triplane at obj_idx {obj_idx} from CPU to GPU...')
                self.current_obj_idx = obj_idx
                self.current_embeddings = self.embeddings[obj_idx.to('cpu')].to(self.device)
            
            self.current_triplanes = self.current_embeddings(torch.tensor(0, dtype=torch.int64).to(self.device)).view(batch_size, 3, self.channels, self.resolution, self.resolution)
            triplanes = self.current_triplanes
        else:
            triplanes = self.embeddings[obj_idx.to('cpu')](torch.tensor(0, dtype=torch.int64).to(self.device)).view(batch_size, 3, self.channels, self.resolution, self.resolution)
        
        # Use tanh to clamp triplanes
        if self.use_tanh:
            triplanes = torch.tanh(triplanes)

        # Triplane aggregating fn.
        
        if self.mode == 'nerf':
            coordinates = coordinates.reshape(coordinates.shape[0], -1, coordinates.shape[-1])  # e.g. [1, 4187904, 3]=

        # TODO: Make sure all these coordinates line up.
        xy_embed = self.sample_plane(coordinates[..., 0:2], triplanes[:, 0])  # ex: [batch_size, 20000, 64]
        yz_embed = self.sample_plane(coordinates[..., 1:3], triplanes[:, 1])
        xz_embed = self.sample_plane(coordinates[..., :3:2], triplanes[:, 2])
        # (M, C)
        
        # aggregate - product or sum?
        if self.aggregate_fn == 'sum':
            features = torch.sum(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0)
        else:
            features = torch.prod(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0)
        # (M, C)

        # decoder
        if self.mode == 'shape':
            return self.net(features)
        elif self.mode == 'nerf':
            if self.view_embedding:
                view_feats = self.view_embedder(-ray_d)[:,None].repeat(1, n_coords, 1).view(batch_size, -1, self.view_embed_dim)  # e.g. [4194304, 27]
                # Concat view embedding to existing feature embedding
                # print(f'features.shape, view_feats.shape: {features.shape}, {view_feats.shape}')  # e.g. torch.Size([1, 4187904, 4]), torch.Size([4187904, 27])
                features = torch.cat([features, view_feats], dim=-1)
            x = self.net(features)
            x = x.reshape(batch_size, ray_batch_size, n_coords, -1)
            rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF, like EG3D
            sigma = x[..., 0:1]
            return {'rgb': rgb, 'sigma': sigma}
    


# For single-scene fitting
class CartesianPlaneNonSirenEmbeddingNetwork(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, aggregate_fn='prod'):  # vs sum
        super().__init__()

        self.aggregate_fn = aggregate_fn
        print(f'Using aggregate_fn {aggregate_fn}')

        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(1, 64, 32, 32)*0.1) for _ in range(3)])

        self.net = nn.Sequential(
            # https://arxiv.org/abs/2006.10739

            nn.Linear(64, 128),
            nn.ReLU(),
            
            nn.Linear(128, 128),
            nn.ReLU(),
            
            nn.Linear(128, output_dim),
        )
        
        self.net.apply(frequency_init(30))
        self.net[0].apply(first_layer_sine_init)

    def sample_plane(self, coords2d, plane):
        assert len(coords2d.shape) == 3, coords2d.shape
        sampled_features = torch.nn.functional.grid_sample(plane,
                                                           coords2d.reshape(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
                                                           mode='bilinear', padding_mode='zeros', align_corners=True)
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.reshape(N, C, H*W).permute(0, 2, 1)
        return sampled_features

    def forward(self, coordinates, debug=False):
        batch_size, n_coords, n_dims = coordinates.shape
        
        xy_embed = self.sample_plane(coordinates[..., 0:2], self.embeddings[0])  # ex: [1, 20000, 64]
        yz_embed = self.sample_plane(coordinates[..., 1:3], self.embeddings[1])
        xz_embed = self.sample_plane(coordinates[..., :3:2], self.embeddings[2])
        # (M, C)
        
        # aggregate - product or sum?
        if self.aggregate_fn == 'sum':
            features = torch.sum(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0)
        else:
            features = torch.prod(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0)
        # (M, C)

        # decoder
        return self.net(features)
    