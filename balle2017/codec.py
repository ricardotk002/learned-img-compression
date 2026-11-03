"""Versioned reference container, separate from the paper's 49-bit header."""
import hashlib
import json
import struct
import zlib
import numpy as np
import torch
from .arithmetic import encode_latents, decode_latents
from .model import Balle2017, round_latents

MAGIC = b'B17REF01'


def fingerprint(model):
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_checkpoint(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = Balle2017(**saved['config'])
    model.load_state_dict(saved['model'])
    return model.eval(), saved


@torch.no_grad()
def compress(model, x, lmbda):
    if model.training or x.shape[0] != 1 or x.device.type != 'cpu':
        raise ValueError('Coding requires one image, CPU, eval mode.')
    if not all(1 <= n <= 65535 for n in x.shape[-2:]):
        raise ValueError('Image dimensions must fit unsigned 16-bit integers.')
    if not 1 <= lmbda <= 65535 or int(lmbda) != lmbda:
        raise ValueError('Lambda must be a positive 16-bit integer.')
    q = round_latents(model.encode(x))
    if q.abs().max() >= 2**30:
        raise ValueError('Latent magnitude exceeds reference coder limit.')
    integers, probs = model.density.discrete_probabilities()
    payload = encode_latents(q[0].to(torch.int32).numpy(), integers, probs)
    metadata = {'size': list(x.shape[-2:]), 'shape': list(q.shape[1:]),
                'lambda': int(lmbda), 'config': model.config, 'model_sha256': fingerprint(model),
                'coder': 'adaptive-binary-reference-v1-not-CABAC',
                'payload_crc32': zlib.crc32(payload)}
    header = json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode()
    return MAGIC + struct.pack('>I', len(header)) + header + payload


def unpack(data):
    if data[:8] != MAGIC or len(data) < 12:
        raise ValueError('Not a Balle2017 reference stream.')
    length = struct.unpack('>I', data[8:12])[0]
    if length > 65536 or len(data) <= 12+length:
        raise ValueError('Truncated or invalid header.')
    metadata = json.loads(data[12:12+length])
    payload = data[12+length:]
    if zlib.crc32(payload) != metadata['payload_crc32']:
        raise ValueError('Corrupt or truncated arithmetic payload.')
    return metadata, payload


@torch.no_grad()
def decompress(model, data):
    metadata, payload = unpack(data)
    if metadata['model_sha256'] != fingerprint(model) or metadata['config'] != model.config:
        raise ValueError('Bitstream requires a different checkpoint.')
    h, w = metadata['size']
    expected = [model.config['channels'], (h+15)//16+1, (w+15)//16+1]
    if not (1 <= h <= 65535 and 1 <= w <= 65535) or metadata['shape'] != expected:
        raise ValueError('Invalid image/latent dimensions.')
    integers, probs = model.density.discrete_probabilities()
    q = decode_latents(payload, expected, integers, probs)
    return model.decode(torch.from_numpy(q).float()[None], (h, w)), metadata
