import math

from pytorch_msssim import ms_ssim


def image_metrics(reference, reconstruction):
    """Full-image RGB metrics on float reconstructions clamped by the codec."""
    mse = (reference - reconstruction).square().mean().item()
    psnr = -10 * math.log10(mse) if mse > 0 else None
    similarity = None
    if min(reference.shape[-2:]) > 160:
        similarity = ms_ssim(reference, reconstruction, data_range=1, size_average=True).item()
    return {"mse": mse, "psnr": psnr, "ms_ssim": similarity}
