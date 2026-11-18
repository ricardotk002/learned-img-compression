"""Full-image RGB evaluation using actual CPU/rANS files and explicit provenance."""

import argparse
import itertools
import statistics
from pathlib import Path

import pandas as pd
import torch

from .bitstream import unpack
from .codec import Codec
from .data import pad_image, read_image, read_manifest, save_image
from .metrics import image_metrics
from .runtime import (
    checkpoint_input_path, configure_cache, environment, input_path, output_path,
    portable_path, sha256, write_json,
)


def mean_defined(records, key):
    values = [row[key] for row in records if row[key] is not None]
    return statistics.mean(values) if values else None


@torch.no_grad()
def evaluate(checkpoints, manifest, output, qualities=None, deltas=(1.0,), labels=None,
             save_recon=False, save_bitstreams=False, repeats=1, threads=4, expected_type=None):
    if repeats < 1 or threads < 1 or not deltas or len(set(deltas)) != len(deltas):
        raise ValueError("repeats/threads must be positive; deltas must be nonempty and unique")
    if labels is not None and len(labels) != len(checkpoints):
        raise ValueError("Supply one label per checkpoint")
    torch.set_num_threads(threads)
    manifest = input_path(manifest, expected_type or "baseline")
    files = read_manifest(manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, points = [], []
    for index, checkpoint in enumerate(checkpoints):
        checkpoint = input_path(checkpoint, expected_type or "baseline")
        codec = Codec(checkpoint)
        kind = codec.model.model_config["type"]
        if expected_type is not None and kind != expected_type:
            raise ValueError("Checkpoint belongs to the other method")
        configure_cache(kind)
        config = codec.checkpoint["config"]
        for split in ("train_manifest", "val_manifest"):
            if split not in config:
                continue
            source = checkpoint_input_path(config[split], kind, config)
            if source.is_file() and set(files) & set(read_manifest(source)):
                raise ValueError(f"Evaluation images overlap {split}: {source}")
        label = labels[index] if labels else kind
        selected = list(range(codec.model.levels)) if qualities is None else list(qualities)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("qualities must be nonempty and unique")
        identifier = f"{checkpoint.parent.name}-{checkpoint.stem}-{codec.checkpoint_hash[:8]}"
        first = read_image(files[0])
        for quality, delta in itertools.product(selected, deltas):
            codec.model.controls(quality, delta)
            warmup, _ = codec.encode(first, quality, delta)
            codec.decode(warmup)
            records = []
            for image_index, path in enumerate(files):
                x = read_image(path).unsqueeze(0)
                height, width = x.shape[-2:]
                encode_times, decode_times = [], []
                for _ in range(repeats):
                    blob, encode_seconds = codec.encode(x, quality, delta)
                    reconstruction, decode_seconds = codec.decode(blob)
                    encode_times.append(encode_seconds)
                    decode_times.append(decode_seconds)
                header, _ = unpack(blob)
                inference = codec.model(pad_image(x), quality, delta)
                expected = inference["x_hat"][..., :height, :width].clamp(0, 1)
                torch.testing.assert_close(reconstruction, expected, rtol=0, atol=1e-6)
                estimated = {name: -torch.log2(p.clamp_min(1e-9)).sum().item() / (height * width)
                             for name, p in inference["likelihoods"].items()}
                y_bytes, z_bytes = header["lengths"]
                header_bytes = len(blob) - y_bytes - z_bytes
                record = {"method": label, "checkpoint": portable_path(checkpoint),
                          "checkpoint_sha256": codec.checkpoint_hash, "image": portable_path(path),
                          "quality": quality, "lambda": config["lambdas"][quality], "delta": delta,
                          "height": height, "width": width, "file_bytes": len(blob),
                          "file_bpp": len(blob) * 8 / (height * width),
                          "payload_bpp": (y_bytes + z_bytes) * 8 / (height * width),
                          "y_bpp": y_bytes * 8 / (height * width),
                          "z_bpp": z_bytes * 8 / (height * width),
                          "header_bpp": header_bytes * 8 / (height * width),
                          "estimated_bpp": sum(estimated.values()),
                          "estimated_y_bpp": estimated["y"], "estimated_z_bpp": estimated["z"],
                          "encode_seconds": statistics.median(encode_times),
                          "decode_seconds": statistics.median(decode_times),
                          **image_metrics(x, reconstruction)}
                stem = f"{image_index:04d}-{path.stem}"
                folder = output / identifier / f"q{quality}-d{delta:g}"
                if save_recon:
                    save_image(reconstruction, folder / f"{stem}.png")
                if save_bitstreams:
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / f"{stem}.lic").write_bytes(blob)
                records.append(record)
                rows.append(record)
            point = {key: records[0][key] for key in (
                "method", "checkpoint", "checkpoint_sha256", "quality", "lambda", "delta"
            )}
            average_keys = ("file_bpp", "payload_bpp", "y_bpp", "z_bpp", "header_bpp",
                            "estimated_bpp", "mse", "psnr", "ms_ssim", "encode_seconds", "decode_seconds")
            point.update({key: mean_defined(records, key) for key in average_keys})
            point.update({"images": len(records), "training_step": codec.checkpoint.get("step"),
                          "phase": codec.checkpoint.get("phase"),
                          "pixel_weighted_file_bpp": sum(row["file_bytes"] for row in records) * 8 /
                          sum(row["height"] * row["width"] for row in records),
                          "ms_ssim_images": sum(row["ms_ssim"] is not None for row in records),
                          "parameters": sum(p.numel() for p in codec.model.parameters()),
                          "parameter_bytes": sum(p.numel() * p.element_size()
                                                 for p in codec.model.parameters()),
                          "checkpoint_file_bytes": checkpoint.stat().st_size})
            points.append(point)
            print(f"{label} q={quality} delta={delta:g}: {point['file_bpp']:.4f} bpp, "
                  f"PSNR={point['psnr']}", flush=True)
        del codec
    pd.DataFrame(rows).to_csv(output / "per_image.csv", index=False)
    pd.DataFrame(points).to_csv(output / "summary.csv", index=False)
    write_json(output / "summary.json", {"metadata": {
        **environment(), "manifest": portable_path(manifest), "manifest_sha256": sha256(manifest),
        "device": "cpu", "threads": threads, "timing_repeats": repeats,
        "rate": "mean per-image total file bpp, including header and both streams",
        "distortion": "RGB [0,1], decoded float reconstruction clamped to [0,1]; no edge crop",
        "ms_ssim": "standard five scales, win_size=11; null if min(H,W)<=160",
        "timing": "median repeats per image then dataset mean; warmed-up CPU; no disk I/O",
        "coding_check": "decoded float reconstruction matches deterministic forward within 1e-6"},
        "points": points})
    return points


@torch.no_grad()
def verify_files(directory, threads=4, expected_type=None):
    """Independently decode existing files; verify forward output and recorded distortion."""
    torch.set_num_threads(threads)
    directory = Path(directory)
    frame = pd.read_csv(directory / "per_image.csv")
    maximum_error, maximum_mse_error, maximum_psnr_error = 0.0, 0.0, 0.0
    hashes = []
    for checkpoint, group in frame.groupby("checkpoint", sort=False):
        codec = Codec(input_path(checkpoint, expected_type or "baseline"))
        if expected_type and codec.model.model_config["type"] != expected_type:
            raise ValueError("Checkpoint belongs to the other method")
        hashes.append(codec.checkpoint_hash)
        for (quality, delta), setting in group.groupby(["quality", "delta"], sort=False):
            folders = list(directory.glob(
                f"*-{Path(checkpoint).stem}-{codec.checkpoint_hash[:8]}/q{quality}-d{delta:g}"
            ))
            if len(folders) != 1:
                raise ValueError(f"Expected one saved-stream directory for q={quality}, delta={delta}")
            for index, row in enumerate(setting.itertuples()):
                original = read_image(input_path(row.image)).unsqueeze(0)
                stream = folders[0] / f"{index:04d}-{Path(row.image).stem}.lic"
                decoded, _ = codec.decode(stream.read_bytes())
                expected = codec.model(pad_image(original), int(quality), float(delta))["x_hat"]
                expected = expected[..., :original.shape[-2], :original.shape[-1]].clamp(0, 1)
                error = (decoded - expected).abs().max().item()
                torch.testing.assert_close(decoded, expected, rtol=0, atol=1e-6)
                maximum_error = max(maximum_error, error)
                metrics = image_metrics(original, decoded)
                mse_error = abs(metrics["mse"] - row.mse)
                psnr_error = abs(metrics["psnr"] - row.psnr) if metrics["psnr"] is not None else 0
                if mse_error > 1e-7 or psnr_error > 1e-4:
                    raise AssertionError(f"Stored metrics changed for {row.image}")
                maximum_mse_error = max(maximum_mse_error, mse_error)
                maximum_psnr_error = max(maximum_psnr_error, psnr_error)
    result = {"files": len(frame), "all_saved_file_round_trips_match_forward": True,
              "maximum_absolute_reconstruction_difference": maximum_error,
              "maximum_logged_mse_difference": maximum_mse_error,
              "maximum_logged_psnr_difference": maximum_psnr_error,
              "checkpoint_sha256": hashes, "environment": environment()}
    write_json(directory / "verification.json", result)
    return result


def main(expected_type="baseline"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+")
    parser.add_argument("--manifest", "--test-manifest", dest="manifest")
    parser.add_argument("--output")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--qualities", nargs="+", type=int)
    parser.add_argument("--deltas", nargs="+", type=float, default=[1.0])
    parser.add_argument("--save-recon", action="store_true")
    parser.add_argument("--save-bitstreams", action="store_true")
    parser.add_argument("--verify-files", action="store_true")
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    configure_cache(expected_type)
    target = output_path(args.output, expected_type, "results/evaluation")
    if args.verify_files:
        print(verify_files(target, args.threads, expected_type))
    else:
        if not args.checkpoints or not args.manifest:
            parser.error("Evaluation requires --checkpoints and --manifest")
        evaluate(args.checkpoints, args.manifest, target, args.qualities, args.deltas,
                 args.labels, args.save_recon, args.save_bitstreams, args.timing_repeats,
                 args.threads, expected_type)


if __name__ == "__main__":
    main()
