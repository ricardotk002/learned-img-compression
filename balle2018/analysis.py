"""Nondominated RD frontiers and overlap-only PCHIP Bjontegaard rate integration."""

import math
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator


def frontier(rates, qualities):
    pairs = sorted((float(rate), float(quality)) for rate, quality in zip(rates, qualities)
                   if np.isfinite(rate) and np.isfinite(quality) and rate > 0)
    best_at_rate = {}
    for rate, quality in pairs:
        best_at_rate[rate] = max(quality, best_at_rate.get(rate, -math.inf))
    result, best_quality = [], -math.inf
    for rate, quality in sorted(best_at_rate.items()):
        if quality > best_quality:
            result.append((rate, quality))
            best_quality = quality
    return np.asarray(result, dtype=float).reshape(-1, 2)


def bd_rate(anchor_rates, anchor_quality, test_rates, test_quality):
    anchor = frontier(anchor_rates, anchor_quality)
    test = frontier(test_rates, test_quality)
    if min(len(anchor), len(test)) < 4:
        raise ValueError("BD-rate needs at least four nondominated points per curve")
    lower = max(anchor[0, 1], test[0, 1])
    upper = min(anchor[-1, 1], test[-1, 1])
    if upper <= lower:
        raise ValueError("No overlapping quality range for BD-rate")
    a = PchipInterpolator(anchor[:, 1], np.log(anchor[:, 0]), extrapolate=False)
    b = PchipInterpolator(test[:, 1], np.log(test[:, 0]), extrapolate=False)
    change = (b.integrate(lower, upper) - a.integrate(lower, upper)) / (upper - lower)
    return {"bd_rate_percent": float(100 * np.expm1(change)),
            "quality_min": float(lower), "quality_max": float(upper)}


def plot_rd(frame, output=None, rate="file_bpp", metric="psnr"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, group in frame.groupby("method", sort=False):
        group = group.dropna(subset=[rate, metric])
        if group.empty:
            continue
        points = frontier(group[rate], group[metric])
        ax.scatter(group[rate], group[metric], alpha=0.35)
        ax.plot(points[:, 0], points[:, 1], "o-", label=label)
    ax.set_xlabel("Total file bits per pixel" if rate == "file_bpp" else rate)
    ax.set_ylabel("RGB PSNR (dB)" if metric == "psnr" else "MS-SSIM")
    ax.grid(alpha=0.3)
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
    return fig, ax
