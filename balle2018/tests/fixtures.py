"""Small deterministic images for correctness tests; no development-script imports."""

import json

import numpy as np
from PIL import Image


def make_data(output):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(99)
    for index in range(6):
        height, width = (192, 203) if index == 5 else (64 + 3 * index, 72 + 2 * index)
        yy, xx = np.mgrid[:height, :width]
        array = np.stack([xx / width, yy / height, (np.sin(xx / 9 + index) + 1) / 2], -1)
        array = np.clip(array * 0.85 + rng.random((height, width, 3)) * 0.15, 0, 1)
        Image.fromarray(np.rint(array * 255).astype(np.uint8)).save(output / f"image{index}.png")
    for name, indices in {"train": range(4), "val": [4], "test": [5]}.items():
        (output / f"{name}.json").write_text(json.dumps([f"image{index}.png" for index in indices]))
