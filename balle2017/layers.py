"""Appendix 6.1: DCT filters and projected symmetric GDN parameters."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def dct_basis(size):
    n = torch.arange(size, dtype=torch.float32)
    k = n[:, None]
    basis = torch.cos(math.pi / size * (n + 0.5) * k) * math.sqrt(2 / size)
    basis[0] /= math.sqrt(2)
    return basis


class DCTConv(nn.Module):
    def __init__(self, inputs, outputs, kernel, stride, transpose=False):
        super().__init__()
        self.stride, self.transpose = stride, transpose
        self.register_buffer('basis', dct_basis(kernel))
        shape = (inputs, outputs, kernel, kernel) if transpose else (outputs, inputs, kernel, kernel)
        spatial = torch.randn(shape)
        spatial /= spatial.square().sum((1, 2, 3), keepdim=True).sqrt()
        self.coefficients = nn.Parameter(self.basis @ spatial @ self.basis.T)
        self.bias = nn.Parameter(torch.zeros(outputs))

    def forward(self, x):
        weight = self.basis.T @ self.coefficients @ self.basis
        operation = F.conv_transpose2d if self.transpose else F.conv2d
        return operation(x, weight, self.bias, stride=self.stride)

    @torch.no_grad()
    def project(self):
        # Orthonormal DCT preserves filter norms. Synthesis is normalized per input.
        norm = self.coefficients.square().sum((1, 2, 3), keepdim=True).sqrt()
        self.coefficients.div_(norm.clamp_min(1e-12))


class GDN(nn.Module):
    offset = 2 ** -10
    lower = 2 ** -5

    def __init__(self, channels, inverse=False):
        super().__init__()
        self.inverse = inverse
        self.beta_prime = nn.Parameter(torch.full((channels,), math.sqrt(1 + self.offset)))
        self.gamma_prime = nn.Parameter((torch.eye(channels) * 0.1 + self.offset).sqrt())

    def forward(self, x):
        beta = self.beta_prime.square() - self.offset
        gamma = self.gamma_prime.square() - self.offset
        norm = F.conv2d(x.square(), gamma[:, :, None, None], beta).clamp_min(1e-12).sqrt()
        return x * norm if self.inverse else x / norm

    @torch.no_grad()
    def project(self):
        self.beta_prime.clamp_(min=self.lower)
        self.gamma_prime.clamp_(min=self.lower)
        self.gamma_prime.copy_((self.gamma_prime + self.gamma_prime.T) / 2)
