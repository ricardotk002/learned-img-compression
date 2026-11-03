"""Three-stage transforms with released-MATLAB symmetric/valid boundaries."""
import numpy as np
import torch
from torch import nn
from .layers import DCTConv, GDN
from .density import SplineDensity


def round_latents(y):
    # MATLAB round: half-integers away from zero, unlike torch.round's ties-to-even.
    return y.sign() * (y.abs() + 0.5).floor()


def boundary_padding(height, width):
    totals = [57 + 16 * ((n + 15) // 16) - n for n in (height, width)]
    return tuple((p // 2, p - p // 2) for p in totals)


def symmetric_pad(x):
    # Includes the edge pixel, as in the authors' MATLAB code. F.pad(reflect) does not.
    indices = []
    for size, (before, after) in zip(x.shape[-2:], boundary_padding(*x.shape[-2:])):
        t = torch.arange(-before, size + after, device=x.device) % (2 * size)
        indices.append(torch.where(t < size, t, 2 * size - 1 - t))
    return x.index_select(-2, indices[0]).index_select(-1, indices[1])


class Balle2017(nn.Module):
    def __init__(self, image_channels=1, channels=128, radius=30):
        super().__init__()
        if image_channels not in (1, 3):
            raise ValueError('Expected grayscale (1) or RGB (3).')
        self.config = dict(image_channels=image_channels, channels=channels, radius=radius)
        self.analysis = nn.ModuleList([
            DCTConv(image_channels, channels, 9, 4), GDN(channels),
            DCTConv(channels, channels, 5, 2), GDN(channels),
            DCTConv(channels, channels, 5, 2), GDN(channels)])
        self.analysis_center = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.synthesis_center = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.synthesis = nn.ModuleList([
            GDN(channels, inverse=True), DCTConv(channels, channels, 5, 2, True),
            GDN(channels, inverse=True), DCTConv(channels, channels, 5, 2, True),
            GDN(channels, inverse=True), DCTConv(channels, image_channels, 9, 4, True)])
        self.density = SplineDensity(channels, radius)

    def encode(self, x):
        y = symmetric_pad(x)
        for layer in self.analysis:
            y = layer(y)
        return y + self.analysis_center

    def decode(self, y, size):
        x = y + self.synthesis_center
        for layer in self.synthesis:
            x = layer(x)
        top, left = [(a - b) // 2 for a, b in zip(x.shape[-2:], size)]
        return x[..., top:top + size[0], left:left + size[1]]

    def forward(self, x, adapt_support=False):
        y = self.encode(x)
        y_hat = y + torch.empty_like(y).uniform_(-0.5, 0.5) if self.training else round_latents(y)
        expanded = self.density.expand_to(y_hat.detach().abs().max()) if adapt_support else False
        if expanded:
            self.config['radius'] = self.density.radius
        return {'x_hat': self.decode(y_hat, x.shape[-2:]), 'y_hat': y_hat,
                'likelihoods': self.density(y_hat, detach=True), 'support_expanded': expanded}

    @torch.no_grad()
    def project(self):
        for layer in list(self.analysis) + list(self.synthesis):
            layer.project()


def import_matlab(model, directory):
    """Load only numeric .mat files, never unpickle downloaded model files."""
    from pathlib import Path
    from scipy.io import loadmat
    root = Path(directory)
    with torch.no_grad():
        for prefix, layers, stages in [('analysis', model.analysis, range(3)),
                                        ('synthesis', model.synthesis, range(1, 4))]:
            for i, stage in enumerate(stages):
                p = loadmat(root / f'{prefix}-{stage:02d}.mat')
                conv, norm = (layers[2*i], layers[2*i+1]) if prefix == 'analysis' else (layers[2*i+1], layers[2*i])
                weight = torch.as_tensor(np.asarray(p['h']), dtype=conv.coefficients.dtype)
                conv.coefficients.copy_(conv.basis @ weight @ conv.basis.T)
                conv.bias.copy_(torch.as_tensor(p['c'].reshape(-1)))
                norm.beta_prime.copy_(torch.as_tensor(p['beta'].reshape(-1) + norm.offset).sqrt())
                norm.gamma_prime.copy_(torch.as_tensor(p['gamma'].T.copy() + norm.offset).sqrt())
        for name, filename in [('analysis_center', 'analysis-03.mat'), ('synthesis_center', 'synthesis-00.mat')]:
            getattr(model, name).copy_(torch.as_tensor(loadmat(root / filename)['c'].reshape(1, -1, 1, 1)))
    # Released transform files do not include the trained spline tables.
