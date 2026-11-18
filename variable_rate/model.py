"""Discrete quality-conditioned channel modulation; continuous latent bin size.

This adapts Choi et al. (2019)'s rate-control mechanism, retaining the scale-only
prior and additive-noise training rather than its full autoregressive model.
"""

import math

from torch import nn
from torch.nn import functional as F

from balle2018.model import Hyperprior


class ChannelCondition(nn.Module):
    def __init__(self, levels, channels):
        super().__init__()
        self.scale = nn.Embedding(levels, channels)
        self.bias = nn.Embedding(levels, channels)
        nn.init.constant_(self.scale.weight, math.log(math.expm1(1.0)))
        nn.init.zeros_(self.bias.weight)

    def forward(self, x, quality):
        scale = F.softplus(self.scale.weight[quality])[None, :, None, None]
        bias = self.bias.weight[quality][None, :, None, None]
        return x * scale + bias


class ConditionalHyperprior(Hyperprior):
    def __init__(self, config):
        if config["type"] != "variable":
            raise ValueError("ConditionalHyperprior requires model.type=variable")
        super().__init__(config)
        if not config.get("conditioning", True) and self.levels != 1:
            raise ValueError("Without conditioning only one quality level is supported")
        if config.get("conditioning", True):
            for name in ("g_a", "g_s", "h_a", "h_s"):
                for index, layer in enumerate(getattr(self, name)):
                    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                        self.conditions[f"{name}_{index}"] = ChannelCondition(
                            self.levels, layer.out_channels
                        )
