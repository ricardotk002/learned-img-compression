"""CPU reference bitstream with separate hyperlatent and main-latent streams."""
from functools import lru_cache
import math
import struct
import zlib

import torch

from balle2017.arithmetic import Arithmetic
from balle2017.codec import fingerprint
from .model import GROUPS


MAGIC = b'ELICSM01'
HEADER = struct.Struct('>8sIIII32sI')
LIMIT = 32


@lru_cache(maxsize=8192)
def counts(index, lower, middle, upper, logistic):
    scale = 2**(index/16)

    def cdf(edge):
        if edge == -LIMIT-1:
            return 0.
        if edge == LIMIT:
            return 1.
        x = (edge+0.5)/scale
        if logistic:
            exp = math.exp(-abs(x))
            return 1/(1+exp) if x >= 0 else exp/(1+exp)
        return math.erfc(-x/math.sqrt(2))/2

    # Reflect positive intervals to preserve precision in Gaussian tails.
    if lower >= 0:
        probability = 1-counts(index, -upper, -middle-1, -lower, logistic)[0]/32768
    else:
        start, split, end = cdf(lower-1), cdf(middle), cdf(upper)
        probability = (split-start)/(end-start) if end > start else 0.5
    left = min(32767, max(1, round(probability*32768)))
    return left, 32768-left


def code_values(coder, mean, scale, values=None, logistic=False):
    indexes = (scale.expand_as(mean).log2()*16).round().clamp(-64, 128).int().flatten().tolist()
    symbols = (values-mean).round().int().flatten().tolist() if values is not None else None
    output = []
    for i, index in enumerate(indexes):
        symbol = symbols[i] if symbols is not None else None
        lower, upper = -LIMIT, LIMIT
        while lower < upper:
            middle = (lower+upper)//2
            bit = None if symbol is None else int(symbol > middle)
            if coder.binary(counts(index, lower, middle, upper, logistic), bit, adaptive=False):
                lower = middle+1
            else:
                upper = middle
        if abs(lower) == LIMIT:
            extra = coder.golomb(None if symbol is None else abs(symbol)-LIMIT)
            lower += extra if lower > 0 else -extra
        output.append(lower)
    return mean.new_tensor(output).reshape_as(mean)+mean


def require_cpu(model):
    if model.training or next(model.parameters()).device.type != 'cpu':
        raise ValueError('File coding requires an evaluation-mode CPU model.')


@torch.no_grad()
def compress(model, x):
    require_cpu(model)
    if x.ndim != 4 or x.shape[:2] != (1, 3) or x.device.type != 'cpu':
        raise ValueError('Expected one RGB CPU image with shape [1, 3, H, W].')
    h, w = x.shape[-2:]
    if not 0 < h*w <= 16_777_216:
        raise ValueError('Image exceeds reference codec size limits.')
    y = model.analyze(x)
    z = model.hyper_analysis(y)
    z_coder = Arithmetic()
    z_hat = code_values(z_coder, model.z_mean.expand_as(z), model.z_scale, z, logistic=True)
    y_coder = Arithmetic()
    groups = y.split(GROUPS, dim=1)
    model.traverse(model.hyper_synthesis(z_hat), lambda i, mask, m, s:
                   code_values(y_coder, m, s, groups[i][..., mask]))
    side, main = z_coder.finish(), y_coder.finish()
    payload = side+main
    return HEADER.pack(MAGIC, h, w, len(side), len(main), bytes.fromhex(fingerprint(model)),
                       zlib.crc32(payload))+payload


@torch.no_grad()
def decompress(model, data):
    require_cpu(model)
    if len(data) < HEADER.size:
        raise ValueError('Truncated ELIC header.')
    magic, h, w, side_size, main_size, identity, crc = HEADER.unpack_from(data)
    payload = data[HEADER.size:]
    if (magic != MAGIC or min(h, w, side_size, main_size) < 1 or h*w > 16_777_216
            or len(payload) != side_size+main_size or zlib.crc32(payload) != crc):
        raise ValueError('Invalid ELIC stream.')
    if identity.hex() != fingerprint(model):
        raise ValueError('Checkpoint does not match the bitstream.')
    z_mean = model.z_mean.expand(1, -1, (h+63)//64, (w+63)//64)
    z_hat = code_values(Arithmetic(payload[:side_size]), z_mean, model.z_scale, logistic=True)
    y_coder = Arithmetic(payload[side_size:])
    y_hat = model.traverse(model.hyper_synthesis(z_hat), lambda _, mask, mean, scale:
                           code_values(y_coder, mean, scale))
    return {'x_hat': model.synthesis(y_hat)[..., :h, :w], 'y_hat': y_hat,
            'header_bytes': HEADER.size, 'z_bytes': side_size, 'y_bytes': main_size}
