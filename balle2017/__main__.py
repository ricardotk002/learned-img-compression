import argparse
import json
from pathlib import Path
import torch
from .codec import compress, decompress, load_checkpoint
from .data import load_image, save_image


def main():
    parser = argparse.ArgumentParser(description='Ballé 2017 reference codec (not original CABAC).')
    parser.add_argument('command', choices=['compress', 'decompress'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model, checkpoint = load_checkpoint(args.checkpoint)
    if not checkpoint.get('density_fitted', False):
        raise ValueError('Checkpoint has no fitted entropy model; train/calibrate it first.')
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if args.command == 'compress':
        x = load_image(args.input, model.config['image_channels'])[None]
        data = compress(model, x, checkpoint['lambda'])
        target.write_bytes(data)
        print(json.dumps({'bytes': len(data), 'actual_bpp': 8*len(data)/(x.shape[-2]*x.shape[-1]),
                          'coder': 'reference-not-CABAC'}))
    else:
        x, metadata = decompress(model, Path(args.input).read_bytes())
        save_image(x[0], target)
        print(json.dumps(metadata))


if __name__ == '__main__':
    main()
