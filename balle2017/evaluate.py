"""Per-image evaluation with coded-rate measurement and latent round-trip checks."""
import argparse
import csv
import io
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .codec import compress, decompress, load_checkpoint, unpack
from .data import load_image, read_manifest, save_image
from .metrics import quality


def classical_encode(image, method, setting):
    buffer = io.BytesIO()
    if method == 'jpeg':
        image.save(buffer, format='JPEG', quality=int(setting), subsampling=2)
    else:
        image.save(buffer, format='JPEG2000', irreversible=True,
                   quality_mode='rates', quality_layers=[float(setting)])
    return buffer.getvalue()


def classical_decode(data, mode):
    with Image.open(io.BytesIO(data)) as image:
        arr = np.asarray(image.convert(mode), dtype=np.float32).copy()/255
    if arr.ndim == 2:
        arr = arr[..., None]
    return torch.from_numpy(arr).permute(2, 0, 1)[None]


def match_classical(image, method, target_bytes):
    """Find a realizable file size near the neural file; never interpolate metrics."""
    best = None
    lower, upper = (1, 100) if method == 'jpeg' else (1., 1000.)
    for _ in range(12):
        setting = (lower+upper)//2 if method == 'jpeg' else (lower+upper)/2
        data = classical_encode(image, method, setting)
        candidate = (abs(len(data)-target_bytes), data, setting)
        if best is None or candidate[0] < best[0]:
            best = candidate
        if method == 'jpeg':
            if len(data) < target_bytes:
                lower = setting+1
            else:
                upper = setting-1
            if lower > upper:
                break
        else:
            if len(data) > target_bytes:
                lower = setting
            else:
                upper = setting
    return best[1], best[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', nargs='+', required=True)
    p.add_argument('--test-manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--crop-border', type=int, default=8, help='Authors removed eight pixels on EACH edge of Kodak.')
    p.add_argument('--skip-ms-ssim', action='store_true')
    p.add_argument('--jpeg-quality', type=int, nargs='*', default=[])
    p.add_argument('--jpeg2000-rates', type=float, nargs='*', default=[], help='OpenJPEG compression ratios, not target bpp.')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--matched-classical', action='store_true', help='Also search per-image JPEG/JPEG2000 file sizes near learned codec.')
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    images = read_manifest(args.test_manifest)
    rows = []
    image_channels = None
    with torch.no_grad():
        for model_index, checkpoint_path in enumerate(args.checkpoints):
            model, checkpoint = load_checkpoint(checkpoint_path)
            if image_channels is not None and image_channels != model.config['image_channels']:
                raise ValueError('Evaluate grayscale and RGB checkpoints in separate reports.')
            image_channels = model.config['image_channels']
            if not checkpoint.get('density_fitted', False):
                raise ValueError('Checkpoint density not fitted; do not report placeholder rates.')
            train_paths = set(checkpoint.get('provenance', {}).get('preprocessing', {}).get('accepted', []))
            if train_paths & {str(p) for p in images}:
                raise ValueError('Test images overlap the training manifest.')
            model_root = root/f'model-{model_index:02d}-lambda-{checkpoint["lambda"]}'
            model_root.mkdir(exist_ok=True)
            for image_index, path in enumerate(images):
                x = load_image(path, model.config['image_channels'], args.crop_border)[None]
                output = model(x)
                data = compress(model, x, checkpoint['lambda'])
                reconstructed, metadata = decompress(model, data)
                if not torch.equal(output['x_hat'], reconstructed):
                    raise AssertionError('Decoded image differs from rounded forward pass.')
                _, payload = unpack(data)
                pixels = x.shape[-2]*x.shape[-1]
                stem = f'{image_index:03d}-{path.stem}'
                (model_root/f'{stem}.b17').write_bytes(data)
                save_image(reconstructed[0], model_root/f'{stem}.png')
                row = {'image': str(path), 'method': 'balle2017-reference-not-CABAC',
                       'checkpoint': str(checkpoint_path), 'lambda': checkpoint['lambda'],
                       'estimated_bpp': float(-output['likelihoods'].log2().sum()/pixels),
                       'actual_bpp': 8*len(data)/pixels, 'payload_bpp': 8*len(payload)/pixels,
                       **quality(x, reconstructed, not args.skip_ms_ssim)}
                rows.append(row)
                print(json.dumps(row), flush=True)
                # Persist partial results during the slow pure-Python coding pass.
                with (root/'progress.jsonl').open('a', encoding='utf-8') as progress:
                    progress.write(json.dumps(row)+'\n')
                # Classical comparisons once, using the same image mode and crop.
                if model_index == 0:
                    array = (x[0].permute(1, 2, 0).numpy()*255+0.5).clip(0, 255).astype(np.uint8)
                    image = Image.fromarray(array[..., 0] if array.shape[-1] == 1 else array)
                    settings = [('jpeg', q) for q in args.jpeg_quality] + [('jpeg2000', r) for r in args.jpeg2000_rates]
                    for method, setting in settings:
                        encoded = classical_encode(image, method, setting)
                        rec = classical_decode(encoded, image.mode)
                        rows.append({'image': str(path), 'method': method, 'setting': setting,
                                     'actual_bpp': 8*len(encoded)/pixels,
                                     **quality(x, rec, not args.skip_ms_ssim)})
                if args.matched_classical:
                    if model_index != 0:
                        array = (x[0].permute(1, 2, 0).numpy()*255+0.5).clip(0, 255).astype(np.uint8)
                        image = Image.fromarray(array[..., 0] if array.shape[-1] == 1 else array)
                    for method in ('jpeg', 'jpeg2000'):
                        encoded, chosen = match_classical(image, method, len(data))
                        rec = classical_decode(encoded, image.mode)
                        target = model_root/f'{method}-matched'
                        target.mkdir(exist_ok=True)
                        extension = 'jpg' if method == 'jpeg' else 'jp2'
                        (target/f'{stem}.{extension}').write_bytes(encoded)
                        save_image(rec[0], target/f'{stem}.png')
                        rows.append({'image': str(path), 'method': method+'-matched',
                            'checkpoint': str(checkpoint_path), 'setting': 'matched-per-image', 'actual_setting': chosen,
                            'actual_bpp': 8*len(encoded)/pixels, 'target_bpp': 8*len(data)/pixels,
                            'rate_mismatch_percent': 100*(len(encoded)/len(data)-1),
                            **quality(x, rec, not args.skip_ms_ssim)})
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with (root/'per_image.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    for row in rows:
        key = (row['method'], row.get('checkpoint', ''), row.get('setting', ''))
        groups.setdefault(key, []).append(row)
    summary = []
    for (method, checkpoint, setting), group in groups.items():
        means = {key: float(np.mean([r[key] for r in group])) for key in group[0]
                 if key not in ('lambda', 'setting') and isinstance(group[0][key], (int, float))}
        summary.append({'method': method, 'checkpoint': checkpoint, 'setting': setting,
                        'images': len(group), **means})
    (root/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (root/'protocol.json').write_text(json.dumps({'arguments': vars(args),
        'metric_input': 'raw floating reconstruction; PNGs are clipped/rounded previews',
        'color': 'full-range JPEG YCbCr, no offset needed for squared differences',
        'ms_ssim': 'pytorch-msssim, luma, data_range=1; historical numerical parity unverified',
        'aggregation': 'arithmetic mean of per-image metrics; fixed lambda/quality for curves; per-image settings for matched comparisons',
        'classical_input': 'same crop/mode, rounded to 8-bit grayscale for JPEG/JPEG2000; error measured against original float luma',
        'matched_rate': 'per-image search for closest realizable classical file bytes; actual rate mismatch reported; no metric interpolation',
        'rate': 'actual_bpp includes full custom container; payload_bpp excludes header',
        'warning': 'Reference arithmetic engine is not original CABAC; see FIDELITY.md'}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
