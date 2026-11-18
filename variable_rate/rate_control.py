"""Per-image monotonicity diagnostics and measured target-bitrate experiments."""

import argparse
import statistics
import time

import pandas as pd
import torch

from balle2018.codec import Codec
from balle2018.data import read_image, read_manifest
from balle2018.metrics import image_metrics
from balle2018.runtime import (
    checkpoint_input_path, configure_cache, environment, input_path, output_path,
    portable_path, sha256, write_json,
)


def diagnose(frame):
    checks = []
    for keys, group in frame.groupby(["checkpoint_sha256", "image", "quality"]):
        group = group.sort_values("delta")
        changes = group.file_bpp.diff().dropna()
        checks.append({"control": "delta", "checkpoint_sha256": keys[0], "image": keys[1],
                       "quality": int(keys[2]), "comparisons": len(changes),
                       "violations": int((changes > 1e-9).sum()),
                       "min_bpp": float(group.file_bpp.min()), "max_bpp": float(group.file_bpp.max())})
    for keys, group in frame.groupby(["checkpoint_sha256", "image", "delta"]):
        group = group.sort_values("quality")
        changes = group.file_bpp.diff().dropna()
        checks.append({"control": "quality", "checkpoint_sha256": keys[0], "image": keys[1],
                       "delta": float(keys[2]), "comparisons": len(changes),
                       "violations": int((changes < -1e-9).sum()),
                       "min_bpp": float(group.file_bpp.min()), "max_bpp": float(group.file_bpp.max())})
    return {"note": "Increasing delta should not increase rate; increasing quality should not "
                    "decrease rate. Equal rates count as non-reversals. Total rate includes headers.",
            "comparisons": sum(row["comparisons"] for row in checks),
            "violations": sum(row["violations"] for row in checks), "checks": checks}


def grid_targets(frame, targets):
    records = []
    for target in targets:
        chosen = []
        for image, group in frame.groupby("image"):
            row = group.loc[(group.file_bpp - target).abs().idxmin()]
            chosen.append({"image": image, "quality": int(row.quality), "delta": float(row.delta),
                           "bpp": float(row.file_bpp), "psnr": float(row.psnr),
                           "error_bpp": float(row.file_bpp - target),
                           "outside_grid_range": not group.file_bpp.min() <= target <= group.file_bpp.max()})
        records.append({"target_bpp": target, "mean_absolute_error_bpp":
                        statistics.mean(abs(row["error_bpp"]) for row in chosen), "images": chosen})
    return {"note": "Closest total-file rate in the evaluated grid, selected by bytes only; "
                    "not a refined target-rate search", "targets": records}


def target_experiment(checkpoint, manifest, targets, output, iterations=8, threads=4):
    torch.set_num_threads(threads)
    codec = Codec(checkpoint)
    files = read_manifest(manifest)
    for key in ("train_manifest", "val_manifest"):
        config = codec.checkpoint["config"]
        if key in config:
            source = checkpoint_input_path(config[key], "variable", config)
            if source.is_file() and set(files) & set(read_manifest(source)):
                raise ValueError(f"Evaluation images overlap {key}")
    rows = []
    for path in files:
        x = read_image(path).unsqueeze(0)
        for target in targets:
            start = time.perf_counter()
            result = codec.search(x, target, iterations=iterations)
            blob = result.pop("blob")
            decoded, _ = codec.decode(blob)
            row = {"image": portable_path(path), **result, **image_metrics(x, decoded),
                   "relative_error_percent": result["error_bpp"] / target * 100,
                   "search_seconds": time.perf_counter() - start}
            rows.append(row)
            print({key: row[key] for key in ("image", "target_bpp", "bpp", "error_bpp")}, flush=True)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "targets-per-image.csv", index=False)
    summaries = []
    for target in targets:
        selected = [row for row in rows if row["target_bpp"] == target]
        summaries.append({"target_bpp": target,
                          "mean_absolute_error_bpp": statistics.mean(abs(row["error_bpp"]) for row in selected),
                          "mean_absolute_relative_error_percent": statistics.mean(
                              abs(row["relative_error_percent"]) for row in selected),
                          "outside_tested_range_images": sum(row["outside_tested_range"] for row in selected),
                          "images": len(selected)})
    write_json(output / "targets-summary.json", {"metadata": {
        **environment(), "checkpoint_sha256": codec.checkpoint_hash,
        "manifest_sha256": sha256(manifest), "device": "cpu", "delta_bounds": [0.5, 2],
        "note": "Closest measured candidate; finite files have discrete attainable rates"},
        "points": summaries})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    diagnostic = sub.add_parser("diagnose")
    diagnostic.add_argument("--per-image", required=True)
    diagnostic.add_argument("--output")
    targets = sub.add_parser("targets")
    targets.add_argument("--checkpoint", required=True)
    targets.add_argument("--manifest", required=True)
    targets.add_argument("--bpps", nargs="+", type=float, required=True)
    targets.add_argument("--output")
    targets.add_argument("--iterations", type=int, default=8)
    targets.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    configure_cache("variable")
    if args.command == "diagnose":
        result = diagnose(pd.read_csv(input_path(args.per_image, "variable")))
        write_json(output_path(args.output, "variable", "results/rate-control.json"), result)
        print({key: result[key] for key in ("comparisons", "violations")})
    else:
        target_experiment(input_path(args.checkpoint, "variable"), input_path(args.manifest, "variable"),
                          args.bpps, output_path(args.output, "variable", "results/targets"),
                          args.iterations, args.threads)


if __name__ == "__main__":
    main()
