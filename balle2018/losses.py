import torch
from torch.nn import functional as F


def rate_distortion(output, target, lmbda, num_pixels=None):
    """R_y + R_z + lambda * 255^2 * RGB MSE, normalized by original pixels."""
    if num_pixels is None:
        num_pixels = target.shape[0] * target.shape[-2] * target.shape[-1]
    rates = {name: -torch.log2(p.clamp_min(1e-9)).sum() / num_pixels
             for name, p in output["likelihoods"].items()}
    mse = F.mse_loss(output["x_hat"], target)
    bpp = sum(rates.values())
    return {"loss": bpp + lmbda * 255**2 * mse, "mse": mse, "bpp": bpp,
            **{f"{name}_bpp": value for name, value in rates.items()}}
