"""Generate method-local reports and figures from saved evaluation measurements."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .analysis import bd_rate, plot_rd
from .runtime import configure_cache, input_path, output_path, write_json


def summarize(evaluation, training=None, compare=(), kind="baseline", anchor=None):
    configure_cache(kind)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evaluation = Path(evaluation)
    payloads = [json.loads((evaluation / "summary.json").read_text())]
    for path in compare:
        path = Path(path)
        payloads.append(json.loads((path / "summary.json" if path.is_dir() else path).read_text()))
    if len({payload["metadata"]["manifest_sha256"] for payload in payloads}) != 1:
        raise ValueError("Evaluation summaries use different dataset manifests")
    frame = pd.DataFrame(payloads[0]["points"])
    combined = pd.concat([pd.DataFrame(payload["points"]) for payload in payloads], ignore_index=True)
    for metric in ("psnr", "ms_ssim"):
        fig, _ = plot_rd(combined, evaluation / f"rd-{metric}.png", metric=metric)
        plt.close(fig)
    table = "| Method | Quality | Delta | File bpp | PSNR (dB) | MS-SSIM |\n"
    table += "|---|---:|---:|---:|---:|---:|\n"
    for row in combined.itertuples():
        psnr = f"{row.psnr:.4f}" if pd.notna(row.psnr) else "null"
        similarity = f"{row.ms_ssim:.6f}" if pd.notna(row.ms_ssim) else "null"
        table += (f"| {row.method} | {row.quality} | {row.delta:g} | {row.file_bpp:.6f} | "
                  f"{psnr} | {similarity} |\n")
    session = None
    if training:
        training = Path(training)
        if (training / "session.json").exists():
            session = json.loads((training / "session.json").read_text())
        if (training / "train.jsonl").exists():
            logs = pd.read_json(training / "train.jsonl", lines=True)
            validation = pd.read_json(training / "validation.jsonl", lines=True)
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            logs.plot(x="step", y="loss", ax=axes[0], title="Sampled training RD loss")
            for phase, group in validation.groupby("phase", sort=False):
                axes[1].plot(group.step, group.loss, ".-", label=phase)
            axes[1].set(xlabel="Step", ylabel="Validation RD loss", title="Phase-specific validation")
            axes[1].legend()
            logs.plot(x="step", y=["y_bpp", "z_bpp"], ax=axes[2], title="Estimated training rates")
            fig.tight_layout()
            fig.savefig(evaluation / "learning-curves.png", dpi=160)
            plt.close(fig)
    diagnostic_text = ""
    if kind == "variable":
        from variable_rate.rate_control import diagnose, grid_targets

        per_image = pd.read_csv(evaluation / "per_image.csv")
        diagnostics = diagnose(per_image)
        write_json(evaluation / "rate-control.json", diagnostics)
        targets = [0.25, 0.6]
        baseline = combined[combined.method == "baseline"]
        if len(baseline) == 1:
            targets.insert(1, float(baseline.iloc[0].file_bpp))
        write_json(evaluation / "grid-targets.json", grid_targets(per_image, targets))
        diagnostic_text = (f"\n## Rate control\n\n{diagnostics['violations']} rate reversals in "
                           f"{diagnostics['comparisons']} adjacent-control comparisons. "
                           f"Dataset-mean rate coverage: {frame.file_bpp.min():.5f}–"
                           f"{frame.file_bpp.max():.5f} bpp. Equal rates count as non-reversals.\n")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for quality, group in frame.groupby("quality"):
            group = group.sort_values("delta")
            axes[0].plot(group.delta, group.file_bpp, "o-", label=f"quality {quality}")
            axes[1].plot(group.file_bpp, group.psnr, "o-", label=f"quality {quality}")
        axes[0].set(xlabel="Quantization step delta", ylabel="Total file bpp")
        axes[1].set(xlabel="Total file bpp", ylabel="RGB PSNR (dB)")
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend()
        fig.tight_layout()
        fig.savefig(evaluation / "control-sweeps.png", dpi=160)
        plt.close(fig)
    if anchor:
        reference = combined[combined.method == anchor]
        results = {label: bd_rate(reference.file_bpp, reference.psnr, group.file_bpp, group.psnr)
                   for label, group in combined.groupby("method") if label != anchor}
        write_json(evaluation / "bd-rate.json", results)
    metadata = payloads[0]["metadata"]
    training_text = ""
    if session:
        training_text = (f"\n## Training\n\n{session['end_step']} completed updates; "
                         f"{session['elapsed_seconds'] / 60:.2f} minutes in the recorded session. "
                         f"Selected checkpoint steps: {sorted(set(frame.training_step))}.\n")
        if session.get("phase_transitions"):
            training_text += f"\nPhase transitions: `{session['phase_transitions']}`.\n"
    report = ("# " + ("Scale-hyperprior" if kind == "baseline" else "Conditional variable-rate") +
              " evaluation\n\n" +
              f"Manifest: `{metadata['manifest']}`; SHA256 `{metadata['manifest_sha256']}`.\n\n" +
              "Full-image RGB PSNR/MS-SSIM, no edge crop, decoded float output clamped to [0,1]. "
              "Rates include headers and both entropy streams, divided by original image pixels. "
              "Scores average per image. This protocol differs from the repository's cropped-luma "
              "Ballé 2017/ELIC table.\n\n" + table + training_text + diagnostic_text +
              "\n## Interpretation\n\nOnly compare overlapping rate ranges. A single fixed-rate "
              "point cannot establish BD-rate; the optional BD-rate calculation requires four "
              "nondominated points per curve and never extrapolates. Equal training wall time "
              "shared across multiple quality objectives is not equal training per operating point. "
              "Short pilots are not converged reproductions.\n\n" +
              "Check `per_image.csv`, `summary.json`, and the generated figures. "
              "Use the evaluator's `--verify-files` command to check saved bitstreams independently.\n")
    (evaluation / "REPORT.md").write_text(report, encoding="utf-8")
    return report


def main(kind="baseline"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--training-run")
    parser.add_argument("--compare", nargs="*", default=[], help="Other evaluation directories or summary JSONs")
    parser.add_argument("--anchor", help="Optional method label for overlap-only PSNR BD-rate")
    args = parser.parse_args()
    directory = output_path(args.evaluation, kind, "results/evaluation")
    summarize(directory, input_path(args.training_run, kind) if args.training_run else None,
              [input_path(path, kind) for path in args.compare], kind, args.anchor)
    print(f"Report and figures: {directory}")


if __name__ == "__main__":
    main()
