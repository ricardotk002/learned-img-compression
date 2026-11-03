"""Download public datasets and build reproducible, content-disjoint manifests."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tarfile
import time
import urllib.request
from functools import partial
import numpy as np
from PIL import Image

IMAGENETTE = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz'
KODAK = 'https://r0k.us/graphics/kodak/kodak/'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(url, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return target
    partial = target.with_suffix(target.suffix+'.part')
    for attempt in range(4):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            request = urllib.request.Request(url, headers={'User-Agent': 'University-image-compression-reproduction/1.0'})
            if offset:
                request.add_header('Range', f'bytes={offset}-')
            with urllib.request.urlopen(request, timeout=90) as response:
                append = offset > 0 and response.status == 206
                if not append:
                    offset = 0
                length = response.headers.get('Content-Length')
                total = offset+int(length) if length else None
                received, last_report = offset, time.monotonic()
                with partial.open('ab' if append else 'wb') as stream:
                    while block := response.read(1024*1024):
                        stream.write(block)
                        received += len(block)
                        if time.monotonic()-last_report > 20:
                            print(json.dumps({'download': target.name, 'MiB': round(received/2**20),
                                              'total_MiB': round(total/2**20) if total else None}), flush=True)
                            last_report = time.monotonic()
                if total and received != total:
                    raise IOError('Incomplete download')
            partial.replace(target)
            print(f'Downloaded {target.name}', flush=True)
            return target
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def extract(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:gz') as stream:
        for member in stream:
            if not (member.isfile() or member.isdir()):
                continue
            path = (destination/member.name).resolve()
            if not path.is_relative_to(destination):
                raise ValueError('Archive path escapes data directory')
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(member) as source, path.open('wb') as target:
                    shutil.copyfileobj(source, target)


def inspect(path, patch=256, saturation_fraction=.01):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert('RGB'))
    h, w = rgb.shape[:2]
    saturation = float(((rgb == 0) | (rgb == 255)).mean())
    reason = 'too_small' if min(h, w)*.75 <= patch else 'saturated' if saturation > saturation_fraction else None
    return {'path': str(path), 'width': w, 'height': h, 'saturated_fraction': saturation,
            'sha256': sha256(path), 'reason': reason}


def prepare(root, train_count, validation_count, seed, saturation_fraction=.05):
    root = Path(root).resolve()
    source = root/'imagenette2'
    if not (source/'train').is_dir():
        raise ValueError('Download/extract full-resolution Imagenette first')
    rng = random.Random(seed)
    records, selected, hashes = {}, {}, set()
    for split, count in [('train', train_count), ('val', validation_count)]:
        paths = sorted((source/split).glob('*/*.JPEG'))
        rng.shuffle(paths)
        with ThreadPoolExecutor(max_workers=4) as pool:
            scanned = list(pool.map(partial(inspect, saturation_fraction=saturation_fraction), paths))
        accepted = []
        for record in scanned:
            if record['sha256'] in hashes:
                record['reason'] = 'duplicate_content'
            elif record['reason'] is None and len(accepted) < count:
                accepted.append(Path(record['path']))
                hashes.add(record['sha256'])
            elif record['reason'] is None:
                record['reason'] = 'not_selected'
        records[split] = scanned
        selected[split] = accepted
        if not accepted:
            raise ValueError(f'No valid images for {split}')
        print(json.dumps({'split': split, 'candidates': len(paths), 'selected': len(accepted), 'requested': count}), flush=True)
    kodak = sorted((root/'kodak').glob('kodim*.png'))
    if len(kodak) != 24:
        raise ValueError('Expected all 24 Kodak test images')
    test_records = []
    for path in kodak:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if sorted(image.size) != [512, 768] or image.mode != 'RGB':
                raise ValueError(f'Unexpected Kodak format: {path}')
        digest = sha256(path)
        if digest in hashes:
            raise ValueError('Kodak/train/validation content overlap')
        hashes.add(digest)
        test_records.append({'path': str(path), 'sha256': digest})
    selected['kodak'] = kodak
    for split, paths in selected.items():
        (root/f'{split}.txt').write_text(''.join(Path(os.path.relpath(p, root)).as_posix()+'\n' for p in paths), encoding='utf-8')
    provenance = {'training_source': IMAGENETTE, 'training_source_description': 'Full-resolution Imagenette v2, ten-class ImageNet subset',
                  'historical_deviation': 'Not the original 6507-image subset; ten-class selection bias',
                  'kodak_source': KODAK, 'seed': seed,
                  'train_count': len(selected['train']), 'validation_count': len(selected['val']), 'test_count': 24,
                  'saturation_fraction': saturation_fraction, 'minimum_size': 'short edge strictly greater than 256/.75',
                  'records': records, 'test_records': test_records,
                  'archive_sha256': sha256(root/'downloads/imagenette2.tgz')}
    (root/'dataset_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in provenance.items() if k not in ('records', 'test_records')}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='data')
    p.add_argument('--download', action='store_true')
    p.add_argument('--train-count', type=int, default=6507)
    p.add_argument('--validation-count', type=int, default=256)
    p.add_argument('--seed', type=int, default=2017)
    p.add_argument('--saturation-fraction', type=float, default=.05)
    args = p.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    if args.download:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda i: download(KODAK+f'kodim{i:02d}.png', root/'kodak'/f'kodim{i:02d}.png'), range(1, 25)))
        archive = download(IMAGENETTE, root/'downloads/imagenette2.tgz')
        marker = root/'imagenette2/.extracted'
        if not marker.exists():
            extract(archive, root)
            marker.write_text(sha256(archive), encoding='ascii')
    prepare(root, args.train_count, args.validation_count, args.seed, args.saturation_fraction)


if __name__ == '__main__':
    main()
