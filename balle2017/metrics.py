"""Full-range JPEG YCbCr, raw float reconstructions, and luma MS-SSIM."""
import math
import torch


def ycbcr(x):
    matrix = x.new_tensor([[.299, .587, .114], [-.168736, -.331264, .5], [.5, -.418688, -.081312]])
    return torch.einsum('ij,bjhw->bihw', matrix, x)


def quality(x, reconstruction, include_msssim=True):
    def psnr(error):
        return -10*math.log10(error) if error > 0 else float('inf')
    mse = float((x-reconstruction).square().mean())
    a, b = (ycbcr(x), ycbcr(reconstruction)) if x.shape[1] == 3 else (x, reconstruction)
    luma = float((a[:, :1]-b[:, :1]).square().mean())
    result = {'mse': mse, 'psnr': psnr(mse), 'luma_psnr': psnr(luma)}
    if x.shape[1] == 3:
        result['chroma_psnr'] = psnr(float((a[:, 1:]-b[:, 1:]).square().mean()))
    if include_msssim:
        from pytorch_msssim import ms_ssim
        # Library convention must be recorded; parity with the 2003 code is pending.
        result['luma_ms_ssim'] = float(ms_ssim(a[:, :1], b[:, :1], data_range=1, size_average=True))
    return result
