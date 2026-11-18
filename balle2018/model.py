"""Scale-only hyperprior, retaining the pilot checkpoint's parameter names.

Transforms and entropy primitives are supplied by the BSD-3-Clause CompressAI
dependency. This project does not vendor its implementation or pretrained weights.
"""

import math

import torch
from compressai.entropy_models import GaussianConditional
from compressai.models import ScaleHyperprior
from torch import nn


class PortableGaussianConditional(GaussianConditional):
    def _standardized_cumulative(self, inputs):
        # Preserve tail precision and gradients: MPS in torch 2.6 lacks erfc.
        values = -inputs / math.sqrt(2)
        if values.device.type == "mps":
            return (0.5 * torch.erfc(values.cpu())).to(inputs.device)
        return 0.5 * torch.erfc(values)


class Hyperprior(ScaleHyperprior):
    """Baseline transforms with common forward/coding paths for both methods."""

    def __init__(self, config):
        super().__init__(N=config["N"], M=config["M"])
        self.gaussian_conditional = PortableGaussianConditional(None)
        self.model_config = dict(config)
        self.levels = config.get("levels", 1)
        if self.levels < 1:
            raise ValueError("At least one quality level is required")
        if config["type"] not in {"baseline", "variable"}:
            raise ValueError("Unknown model type")
        if config["type"] == "baseline" and self.levels != 1:
            raise ValueError("Baseline needs exactly one quality level")
        self.conditions = nn.ModuleDict()

    def controls(self, quality, delta):
        if not isinstance(quality, int) or not 0 <= quality < self.levels:
            raise ValueError(f"quality must be an integer in [0, {self.levels - 1}]")
        if not math.isfinite(delta) or not 0.01 <= delta <= 100:
            raise ValueError("delta must be finite and in [0.01, 100]")

    def transform(self, name, x, quality):
        for index, layer in enumerate(getattr(self, name)):
            x = layer(x)
            key = f"{name}_{index}"
            if key in self.conditions:
                x = self.conditions[key](x, quality)
        return x

    def forward(self, x, quality=0, delta=1.0):
        self.controls(quality, delta)
        y = self.transform("g_a", x, quality)
        z = self.transform("h_a", y.abs(), quality)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales = self.transform("h_s", z_hat, quality)
        scales = self.gaussian_conditional.lower_bound_scale(scales) / delta
        symbols_hat, y_likelihoods = self.gaussian_conditional(y / delta, scales)
        x_hat = self.transform("g_s", symbols_hat * delta, quality)
        return {"x_hat": x_hat, "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}}

    def compress(self, x, quality=0, delta=1.0):
        self.controls(quality, delta)
        y = self.transform("g_a", x, quality)
        z = self.transform("h_a", y.abs(), quality)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.shape[-2:])
        scales = self.transform("h_s", z_hat, quality)
        scales = self.gaussian_conditional.lower_bound_scale(scales) / delta
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_strings = self.gaussian_conditional.compress(y / delta, indexes)
        return {"strings": [y_strings, z_strings], "shape": tuple(z.shape[-2:])}

    def decompress(self, strings, shape, quality=0, delta=1.0):
        self.controls(quality, delta)
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales = self.transform("h_s", z_hat, quality)
        scales = self.gaussian_conditional.lower_bound_scale(scales) / delta
        indexes = self.gaussian_conditional.build_indexes(scales)
        symbols = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        return {"x_hat": self.transform("g_s", symbols * delta, quality).clamp(0, 1)}


def build_model(config):
    if config["type"] == "variable":
        from variable_rate.model import ConditionalHyperprior

        return ConditionalHyperprior(config)
    if config["type"] != "baseline":
        raise ValueError("model.type must be baseline or variable")
    return Hyperprior(config)
