"""Evaluate an RGB ELIC checkpoint using the Ballé metric and codec helpers."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import torch
from PIL import Image

from balle2017.data import load_image, read_manifest, save_image
from balle2017.evaluate import classical_decode, classical_encode, match_classical
from balle2017.metrics import quality
from .codec import compress, decompress
from .model import load_checkpoint


@torch.no_grad()
def evaluate(args):
    torch.set_num_threads(args.threads)
    model, checkpoint = load_checkpoint(args.checkpoint)
    images = read_manifest(args.test_manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    settings = [('jpeg', q) for q in args.jpeg_quality]
    settings += [('jpeg2000', r) for r in args.jpeg2000_rates]
    with (args.output/'progress.jsonl').open('w', encoding='utf-8') as progress:
        for index, path in enumerate(images, start=1):
            x = load_image(path, 3, args.crop_border)[None]
            pixels = x.shape[-2]*x.shape[-1]
            stream = compress(model, x)
            decoded = decompress(model, stream)
            expected = model(x)
            for key in ('x_hat', 'y_hat'):
                if not torch.equal(decoded[key], expected[key]):
                    raise AssertionError(f'{path.name}: decoded {key} differs from inference.')
            root = args.output/path.stem
            root.mkdir(exist_ok=True)
            (root/'image.elic').write_bytes(stream)
            save_image(decoded['x_hat'][0], root/'elic.png')
            save_image(x[0], root/'original.png')
            row = {'image': path.name, 'method': 'elic-full' if model.variant == 'full' else 'elic-sm',
                   'setting': checkpoint['lambda'],
                   'file_bytes': len(stream), 'actual_bpp': 8*len(stream)/pixels,
                   'estimated_bpp': float(expected['y_bpp']+expected['z_bpp']),
                   'payload_bpp': 8*(decoded['y_bytes']+decoded['z_bytes'])/pixels,
                   **{key: decoded[key] for key in ('header_bytes', 'y_bytes', 'z_bytes')},
                   **quality(x, decoded['x_hat'])}
            rows.append(row)
            image = Image.open(root/'original.png')
            with image:
                for method, setting in settings:
                    data = classical_encode(image, method, setting)
                    rows.append(classical_row(x, path.name, pixels, method, setting, data))
                for method, extension in (('jpeg', 'jpg'), ('jpeg2000', 'jp2')):
                    data, setting = match_classical(image, method, len(stream))
                    matched = classical_row(x, path.name, pixels, method+'-matched', 'matched', data)
                    rows.append({**matched, 'actual_setting': setting, 'target_bpp': row['actual_bpp'],
                                 'rate_mismatch_percent': 100*(len(data)/len(stream)-1)})
                    (root/f'{method}.{extension}').write_bytes(data)
                    save_image(classical_decode(data, 'RGB')[0], root/f'{method}.png')
            line = json.dumps({'completed': index, 'total': len(images), **row})
            print(line, flush=True)
            progress.write(line+'\n')
            progress.flush()
    with (args.output/'per_image.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=sorted(set().union(*(row.keys() for row in rows))))
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    for row in rows:
        groups.setdefault((row['method'], row['setting']), []).append(row)
    summary = []
    for (method, setting), group in groups.items():
        metrics = {key: mean(row[key] for row in group) for key, value in group[0].items()
                   if key != 'setting' and isinstance(value, (int, float))}
        if 'rate_mismatch_percent' in metrics:
            errors = [abs(row['rate_mismatch_percent']) for row in group]
            metrics.update(mean_absolute_rate_mismatch_percent=mean(errors),
                           max_absolute_rate_mismatch_percent=max(errors))
        summary.append(dict(method=method, setting=setting, images=len(group), **metrics))
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    protocol = {'checkpoint': str(args.checkpoint), 'step': checkpoint['step'],
                'test_manifest': str(args.test_manifest), 'crop_border': args.crop_border,
                'image_mode': 'RGB', 'variant': model.variant, 'aggregate': 'arithmetic mean of per-image scores',
                'rate': 'whole-file bytes per cropped pixel; payload excludes the header',
                'quality': 'raw RGB model output, RGB PSNR and JPEG luma PSNR/MS-SSIM',
                'classical': 'RGB JPEG 4:2:0 and Pillow OpenJPEG; matched by file size'}
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')


def classical_row(x, name, pixels, method, setting, data):
    return {'image': name, 'method': method, 'setting': setting, 'file_bytes': len(data),
            'actual_bpp': 8*len(data)/pixels, **quality(x, classical_decode(data, 'RGB'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--test-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--crop-border', type=int, default=8)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--jpeg-quality', type=int, nargs='*', default=[10, 20, 30, 40, 50, 70, 90])
    parser.add_argument('--jpeg2000-rates', type=float, nargs='*', default=[8, 10, 12, 16, 20, 30, 40])
    args = parser.parse_args()
    if args.crop_border < 0 or args.threads < 1:
        parser.error('Crop border must be nonnegative and threads must be positive.')
    evaluate(args)


if __name__ == '__main__':
    main()
