"""Compress or decompress a single RGB image with an ELIC checkpoint."""
import argparse
import json
from pathlib import Path

import torch

from balle2017.data import load_image, save_image
from .codec import compress, decompress, HEADER
from .model import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('compress', 'decompress'))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, _ = load_checkpoint(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == 'compress':
        data = compress(model, load_image(args.input, 3)[None])
        args.output.write_bytes(data)
        _, h, w, side, main_bytes, _, _ = HEADER.unpack_from(data)
        print(json.dumps({'z_bytes': side, 'y_bytes': main_bytes, 'header_bytes': HEADER.size,
                          'actual_bpp': 8*len(data)/(h*w)}))
    else:
        image = decompress(model, args.input.read_bytes())['x_hat'][0]
        save_image(image, args.output)


if __name__ == '__main__':
    main()
