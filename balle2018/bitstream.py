"""Original versioned pilot container; preserves compatibility with local .lic files."""

import json
import struct

MAGIC = b"NYUIC01\0"


def pack(encoded, height, width, quality, delta, checkpoint_hash, model_config):
    streams = [encoded["strings"][0][0], encoded["strings"][1][0]]
    header = {"version": 1, "height": height, "width": width,
              "z_shape": list(encoded["shape"]), "quality": quality, "delta": delta,
              "checkpoint_sha256": checkpoint_hash, "model": model_config,
              "coder": "ans", "device": "cpu", "lengths": [len(stream) for stream in streams]}
    metadata = json.dumps(header, separators=(",", ":"), allow_nan=False).encode()
    return MAGIC + struct.pack(">I", len(metadata)) + metadata + b"".join(streams)


def unpack(blob):
    if len(blob) < 12 or blob[:8] != MAGIC:
        raise ValueError("Not a supported NYU image-compression file")
    size = struct.unpack(">I", blob[8:12])[0]
    if size > 65536 or len(blob) < 12 + size:
        raise ValueError("Truncated or oversized bitstream header")
    header = json.loads(blob[12:12 + size])
    if header["version"] != 1 or header["coder"] != "ans" or header["device"] != "cpu":
        raise ValueError("Unsupported bitstream version/coder/device")
    lengths = header["lengths"]
    if len(lengths) != 2 or any(not isinstance(value, int) or value <= 0 for value in lengths):
        raise ValueError("Invalid stream lengths")
    if 12 + size + sum(lengths) != len(blob):
        raise ValueError("Bitstream length mismatch")
    height, width = header["height"], header["width"]
    if not all(isinstance(value, int) and value > 0 for value in (height, width)):
        raise ValueError("Invalid original dimensions")
    if header["z_shape"] != [(height + 63) // 64, (width + 63) // 64]:
        raise ValueError("Hyperlatent shape does not match original dimensions")
    start = 12 + size
    return header, [[blob[start:start + lengths[0]]], [blob[start + lengths[0]:]]]
