"""Per-channel, spatially shared linear splines, ten knots per unit (2017)."""
import torch
import math
from torch import nn


class SplineDensity(nn.Module):
    def __init__(self, channels, radius=30, resolution=10):
        super().__init__()
        self.radius, self.resolution = radius, resolution
        grid = torch.linspace(-radius, radius, 2 * radius * resolution + 1)
        values = torch.exp(-grid.abs()) / 2
        self.values = nn.Parameter(values.repeat(channels, 1))
        self.project()

    def forward(self, y, detach=False):
        if y.detach().abs().max() > self.radius:
            raise ValueError('Latent outside spline support: enlarge --radius; no silent clipping.')
        values = self.values.detach() if detach else self.values
        position = (y + self.radius) * self.resolution
        left = position.floor().long().clamp(0, values.shape[1] - 2)
        fraction = position - left
        channels = torch.arange(y.shape[1], device=y.device)[None, :, None, None]
        return (values[channels, left] * (1 - fraction) + values[channels, left + 1] * fraction).clamp_min(1e-12)

    @torch.no_grad()
    def project(self):
        self.values.clamp_(min=1e-9)
        area = (self.values[:, :-1] + self.values[:, 1:]).sum(1, keepdim=True) / (2 * self.resolution)
        self.values.div_(area)

    @torch.no_grad()
    def expand_to(self, maximum, margin=2):
        """Conservative range heuristic: retain knots, append positive tail knots.

        Reconstructs the unspecified historical heuristic without claiming parity.
        Returns True when the Parameter changes; plain SGD must then be rebound.
        Never shrink, preserving fitted density detail and optimizer-free tails.
        """
        needed = math.ceil(float(maximum) + margin)
        if needed <= self.radius:
            return False
        radius = max(needed, math.ceil(self.radius*1.25))
        if radius > 512:
            raise FloatingPointError('Latent support exceeded 512; inspect training stability.')
        padding = (radius-self.radius)*self.resolution
        values = torch.nn.functional.pad(self.values.detach(), (padding, padding), value=1e-9)
        self.values = nn.Parameter(values)
        self.radius = radius
        self.project()
        return True

    @torch.no_grad()
    def discrete_probabilities(self):
        # Eq. 10: sample the NOISY density at integer locations, not integrate it again.
        integers = torch.arange(-self.radius, self.radius + 1, device=self.values.device)
        indices = ((integers + self.radius) * self.resolution).long()
        probs = self.values[:, indices].double()
        return integers.cpu().numpy(), (probs / probs.sum(1, keepdim=True)).cpu().numpy()
