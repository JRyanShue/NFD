from PIL import Image
from torchvision import transforms
import torch

def get_mgrid(img_size=64):
    x = torch.linspace(-1, 1, img_size)
    y = torch.linspace(-1, 1, img_size)
    xt, yt = torch.meshgrid(x, y)
    coords = torch.stack([yt, xt], -1)
    return coords

def extract_samples(image, crop_left=None, crop_top=None, crop_width=None):
    image = torch.FloatTensor(image)
    if image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    coords = get_mgrid(image.shape[0]).reshape(-1, 2)
    return coords, image.reshape(-1, 3)

class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, img_name, img_size, crop_left=0, crop_top=0, crop_width=None):
        super().__init__()

        self.transform = transforms.Compose(
        [transforms.Resize(img_size), transforms.CenterCrop(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

        img = Image.open(img_name)
        img = self.transform(img)
        if crop_width == None:
            crop_width = img_size
        self.coordinates, self.values = extract_samples(img, crop_left, crop_top, crop_width)


    def __len__(self):
        return self.coordinates.shape[0]

    def __getitem__(self, idx):
        return self.coordinates[idx], self.values[idx]