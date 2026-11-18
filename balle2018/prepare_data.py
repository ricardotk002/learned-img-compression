"""Download natural-image datasets or create deterministic JSON manifests from local images."""

import argparse
import hashlib
import os
import random
import shutil
import time
import zipfile
from pathlib import Path

import requests
from PIL import Image

from .data import EXTENSIONS, read_image, read_manifest
from .runtime import (
    ROOT, configure_cache, input_path, method_name, output_path, portable_path, sha256, write_json,
)

DIV2K_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
KODAK_URL = "https://r0k.us/graphics/kodak/kodak/kodim{:02d}.png"


def prepare(root, output, name, val_fraction=0.05, seed=42, evaluation=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not root.is_dir() or not 0 < val_fraction < 1:
        raise ValueError("Image root must exist; validation fraction must be between 0 and 1")
    files = sorted(path for path in root.rglob("*")
                   if path.is_file() and path.suffix.lower() in EXTENSIONS)
    if not files:
        raise ValueError(f"No images found in {root}")
    unique, duplicates, hashes = [], [], set()
    for path in files:
        image = read_image(path)
        digest = hashlib.sha256(str(tuple(image.shape)).encode() + image.numpy().tobytes()).hexdigest()
        if digest in hashes:
            duplicates.append(portable_path(path))
        else:
            hashes.add(digest)
            unique.append(path)
    if evaluation:
        splits = {name: unique}
    else:
        if len(unique) < 2:
            raise ValueError("At least two unique images are required for train/val")
        random.Random(seed).shuffle(unique)
        count = min(len(unique) - 1, max(1, round(len(unique) * val_fraction)))
        splits = {"train": sorted(unique[count:]), "val": sorted(unique[:count])}
    for split in splits:
        if (output / f"{split}.json").exists():
            raise FileExistsError(f"Manifest already exists: {output / f'{split}.json'}")
    for split, paths in splits.items():
        write_json(output / f"{split}.json", [Path(os.path.relpath(path, output)).as_posix()
                                             for path in paths])
    write_json(output / f"{name}-metadata.json", {
        "dataset_name": name, "root": portable_path(root), "seed": seed,
        "validation_fraction": None if evaluation else val_fraction,
        "duplicates_excluded": duplicates,
        "files": {Path(os.path.relpath(path, output)).as_posix(): sha256(path) for path in unique},
        "splits": {key: len(value) for key, value in splits.items()},
    })
    print({key: len(value) for key, value in splits.items()}, flush=True)
    return splits


def center_crops(manifest, directory, size=512):
    directory = Path(directory)
    manifest = Path(manifest)
    files, records = [], []
    for path in read_manifest(manifest):
        target = directory / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as image:
            width, height = image.size
            if min(width, height) < size:
                raise ValueError(f"Image too small for requested crop: {path}")
            left, top = (width - size) // 2, (height - size) // 2
            box = (left, top, left + size, top + size)
            if not target.exists():
                image.convert("RGB").crop(box).save(target)
        files.append(Path(os.path.relpath(target, manifest.parent)).as_posix())
        records.append({"source": portable_path(path), "source_sha256": sha256(path),
                        "crop": portable_path(target), "crop_sha256": sha256(target), "box": box})
    write_json(manifest.parent / "val-crops.json", files)
    write_json(manifest.parent / "validation-crops-metadata.json", {
        "note": f"Fixed {size}x{size} center crops of held-out originals; no resizing",
        "images": records,
    })


def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            with requests.get(url, headers={"Range": f"bytes={offset}-"} if offset else {},
                              stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                resumed = offset > 0 and response.status_code == 206
                if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Unexpected resumed byte range")
                received = offset if resumed else 0
                total = received + int(response.headers.get("Content-Length", 0))
                next_report = received + 128 * 1024**2
                with partial.open("ab" if resumed else "wb") as file:
                    for block in response.iter_content(1024**2):
                        file.write(block)
                        received += len(block)
                        if received >= next_report:
                            print(f"{destination.name}: {received / 1024**2:.0f}/"
                                  f"{total / 1024**2:.0f} MiB", flush=True)
                            next_report = received + 128 * 1024**2
                if total and received != total:
                    raise OSError("Incomplete download")
            partial.replace(destination)
            return
        except (requests.RequestException, OSError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def acquire(data):
    """Official DIV2K HR train set (800 images) and Kodak (24 images)."""
    data = Path(data)
    originals = data / "div2k/DIV2K_train_HR"
    archive = data / "downloads/DIV2K_train_HR.zip"
    if len(list(originals.glob("*.png"))) != 800:
        download(DIV2K_URL, archive)
        destination = (data / "div2k").resolve()
        with zipfile.ZipFile(archive) as source:
            entries = [entry for entry in source.infolist()
                       if entry.filename.startswith("DIV2K_train_HR/") and entry.filename.endswith(".png")]
            if len(entries) != 800:
                raise ValueError("DIV2K archive does not contain 800 originals")
            for entry in entries:
                target = (destination / entry.filename).resolve()
                if not target.is_relative_to(destination):
                    raise ValueError("Archive member escapes dataset directory")
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".png.part")
                with source.open(entry) as incoming, temporary.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                temporary.replace(target)
    kodak = data / "kodak"
    for number in range(1, 25):
        target = kodak / f"kodim{number:02d}.png"
        download(KODAK_URL.format(number), target)
        with Image.open(target) as image:
            image.verify()
        with Image.open(target) as image:
            if sorted(image.size) != [512, 768]:
                raise ValueError(f"Unexpected Kodak dimensions: {target}")
    sources = {"div2k": {"url": DIV2K_URL, "citation": "Agustsson and Timofte, NTIRE 2017 "
                         "Challenge on Single Image Super-Resolution: Dataset and Study"},
               "kodak": {"url_pattern": KODAK_URL,
                          "files": {path.name: sha256(path) for path in sorted(kodak.glob("*.png"))}}}
    if archive.is_file():
        sources["div2k"]["archive_sha256"] = sha256(archive)
    write_json(data / "downloads/sources.json", sources)


def main(kind="baseline"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="Read images from an existing directory recursively")
    parser.add_argument("--output", help="Method-local manifest directory")
    parser.add_argument("--name", default="images")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--center-crop", type=int, default=512)
    parser.add_argument("--evaluation", action="store_true")
    parser.add_argument("--download", action="store_true", help="Download DIV2K HR and Kodak")
    args = parser.parse_args()
    configure_cache(kind)
    manifests = output_path(args.output, kind, "data/manifests/div2k-pilot")
    data = ROOT / method_name(kind) / "data"
    if args.download:
        acquire(data)
        if not (manifests / "train.json").exists():
            prepare(data / "div2k/DIV2K_train_HR", manifests, "div2k-train-hr",
                    args.val_fraction, args.seed)
        if not (manifests / "kodak.json").exists():
            prepare(data / "kodak", manifests, "kodak", evaluation=True)
        if args.center_crop:
            center_crops(manifests / "val.json", data / "div2k/validation-center512", args.center_crop)
    else:
        if not args.root:
            parser.error("Provide --root or --download")
        prepare(input_path(args.root, kind), manifests, args.name,
                args.val_fraction, args.seed, args.evaluation)
        if not args.evaluation and args.center_crop:
            center_crops(manifests / "val.json", data / "div2k/validation-center512", args.center_crop)


if __name__ == "__main__":
    main()
