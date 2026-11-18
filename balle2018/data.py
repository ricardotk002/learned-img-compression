"""RGB [0,1] images, explicit JSON/text manifests, and scale-preserving crops."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torch.utils.data import Dataset

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


def read_manifest(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    entries = json.loads(text) if path.suffix.lower() == ".json" else [
        line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]
    files = [(path.parent / entry).resolve() for entry in entries]
    if not files or len(files) != len(set(files)):
        raise ValueError(f"Manifest must contain nonempty, unique entries: {path}")
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing images: {missing[:5]}")
    return files


def read_image(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        array = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def save_image(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    Image.fromarray(np.rint(array * 255).astype(np.uint8)).save(path)


def pad_image(x, multiple=64):
    height, width = x.shape[-2:]
    return F.pad(x, (0, (-width) % multiple, 0, (-height) % multiple), mode="replicate")


class CropDataset(Dataset):
    def __init__(self, manifest, size):
        self.files = read_manifest(manifest)
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        x = read_image(self.files[index])
        height, width = x.shape[-2:]
        x = F.pad(x, (0, max(0, self.size - width), 0, max(0, self.size - height)),
                  mode="replicate")
        height, width = x.shape[-2:]
        top = random.randrange(height - self.size + 1)
        left = random.randrange(width - self.size + 1)
        x = x[:, top:top + self.size, left:left + self.size]
        return x.flip(-1) if random.random() < 0.5 else x
