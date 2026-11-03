"""Explicit manifests and auditable training preprocessing."""
import math
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset


def load_image(path, channels=1, crop_border=0):
    with Image.open(path) as image:
        array = np.asarray(image.convert('RGB'), dtype=np.float32).copy() / 255
    x = torch.from_numpy(array).permute(2, 0, 1)
    if channels == 1:
        x = (x * x.new_tensor([0.299, 0.587, 0.114])[:, None, None]).sum(0, keepdim=True)
    if crop_border:
        if min(x.shape[-2:]) <= 2*crop_border:
            raise ValueError('Image too small for border crop.')
        x = x[:, crop_border:-crop_border, crop_border:-crop_border]
    return x


def save_image(x, path):
    array = (x.detach().cpu().clamp(0, 1)*255 + 0.5).to(torch.uint8).permute(1, 2, 0).numpy()
    Image.fromarray(array[..., 0] if array.shape[-1] == 1 else array).save(path)


def read_manifest(path):
    path = Path(path).resolve()
    lines = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip() and not line.startswith('#')]
    images = [(path.parent / line).resolve() for line in lines]
    if not images or any(not p.is_file() for p in images):
        raise ValueError('Manifest is empty or contains missing images.')
    if len(set(images)) != len(images):
        raise ValueError('Duplicate image paths in manifest.')
    return images


class TrainingImages(Dataset):
    def __init__(self, manifest, channels=1, patch=256, saturation_fraction=0.01):
        self.images = read_manifest(manifest)
        self.channels, self.patch = channels, patch
        self.accepted, self.rejected = [], []
        for path in self.images:
            x = load_image(path, 3)
            saturated = float(((x <= 0) | (x >= 1)).float().mean())
            if min(x.shape[-2:]) * 0.75 <= patch:
                self.rejected.append((str(path), 'too_small'))
            elif saturated > saturation_fraction:
                self.rejected.append((str(path), 'saturated'))
            else:
                self.accepted.append(path)
        if not self.accepted:
            raise ValueError('No training images survive size/saturation filters.')

    def __len__(self):
        return len(self.accepted)

    def __getitem__(self, index):
        x = load_image(self.accepted[index], 3)
        # Pixel dequantization happens before downsampling. Do not clamp noise away.
        x = x + torch.empty_like(x).uniform_(-0.5/255, 0.5/255)
        low = self.patch / min(x.shape[-2:])
        scale = float(torch.empty(()).uniform_(low, 0.75))
        size = [max(self.patch, math.floor(n*scale)) for n in x.shape[-2:]]
        x = F.interpolate(x[None], size=size, mode='bicubic', align_corners=False, antialias=True)[0]
        top = int(torch.randint(x.shape[-2]-self.patch+1, ()))
        left = int(torch.randint(x.shape[-1]-self.patch+1, ()))
        x = x[:, top:top+self.patch, left:left+self.patch]
        if self.channels == 1:
            x = (x*x.new_tensor([0.299, 0.587, 0.114])[:, None, None]).sum(0, keepdim=True)
        return x

    def report(self):
        return {'accepted': [str(p) for p in self.accepted], 'rejected': self.rejected,
                'warning': 'Saturation threshold, scale distribution, and interpolation are reconstruction choices; original details unavailable.'}
