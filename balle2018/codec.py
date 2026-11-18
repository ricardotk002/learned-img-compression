"""CPU/rANS file coding shared with the conditional hyperprior."""

import argparse
import math
import time

import compressai
import torch

from .bitstream import pack, unpack
from .data import pad_image, read_image, save_image
from .runtime import configure_cache, input_path, load_model, output_path, sha256


class Codec:
    def __init__(self, checkpoint):
        compressai.set_entropy_coder("ans")
        self.checkpoint_hash = sha256(checkpoint)
        self.model, self.checkpoint = load_model(checkpoint, coding=True)

    @torch.no_grad()
    def encode(self, image, quality=0, delta=1.0):
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.shape[0] != 1:
            raise ValueError("File coding supports one image per file")
        image = image.cpu()
        height, width = image.shape[-2:]
        start = time.perf_counter()
        encoded = self.model.compress(pad_image(image), quality, delta)
        blob = pack(encoded, height, width, quality, delta,
                    self.checkpoint_hash, self.model.model_config)
        return blob, time.perf_counter() - start

    @torch.no_grad()
    def decode(self, blob):
        start = time.perf_counter()
        header, strings = unpack(blob)
        if header["checkpoint_sha256"] != self.checkpoint_hash:
            raise ValueError("Bitstream was encoded with a different checkpoint")
        if header["model"] != self.model.model_config:
            raise ValueError("Bitstream model configuration does not match checkpoint")
        result = self.model.decompress(strings, header["z_shape"], header["quality"], header["delta"])
        return (result["x_hat"][..., :header["height"], :header["width"]],
                time.perf_counter() - start)

    def search(self, image, target_bpp, qualities=None, bounds=(0.5, 2.0), iterations=8):
        """Closest actual-file-rate candidate; log-grid probes followed by crossing refinement."""
        if (not math.isfinite(target_bpp) or target_bpp <= 0 or iterations < 1
                or not 0.01 <= bounds[0] <= bounds[1] <= 100):
            raise ValueError("Invalid target, delta bounds, or search iterations")
        qualities = list(range(self.model.levels)) if qualities is None else list(qualities)
        if not qualities:
            raise ValueError("At least one quality level is required")
        pixels = image.shape[-2] * image.shape[-1]
        candidates = []

        def probe(quality, delta):
            blob, _ = self.encode(image, quality, delta)
            candidate = {"quality": quality, "delta": delta, "bpp": len(blob) * 8 / pixels,
                         "blob": blob}
            candidates.append(candidate)
            return candidate

        for quality in qualities:
            grid = [bounds[0] * (bounds[1] / bounds[0]) ** (index / 8) for index in range(9)]
            samples = [probe(quality, delta) for delta in grid]
            for left, right in zip(samples[:-1], samples[1:]):
                if (left["bpp"] - target_bpp) * (right["bpp"] - target_bpp) > 0:
                    continue
                for _ in range(iterations):
                    middle = probe(quality, (left["delta"] * right["delta"]) ** 0.5)
                    if (left["bpp"] - target_bpp) * (middle["bpp"] - target_bpp) <= 0:
                        right = middle
                    else:
                        left = middle
        best = min(candidates, key=lambda candidate: abs(candidate["bpp"] - target_bpp))
        minimum = min(candidate["bpp"] for candidate in candidates)
        maximum = max(candidate["bpp"] for candidate in candidates)
        return {**best, "target_bpp": target_bpp, "error_bpp": best["bpp"] - target_bpp,
                "probes": len(candidates), "tested_min_bpp": minimum, "tested_max_bpp": maximum,
                "outside_tested_range": not minimum <= target_bpp <= maximum}


def main(expected_type="baseline"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["compress", "decompress"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--quality", type=int, default=0)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--target-bpp", type=float)
    args = parser.parse_args()
    configure_cache(expected_type)
    torch.set_num_threads(args.threads)
    codec = Codec(input_path(args.checkpoint, expected_type))
    if codec.model.model_config["type"] != expected_type:
        parser.error("Checkpoint belongs to the other method")
    target = output_path(args.output, expected_type, "results/image.lic")
    target.parent.mkdir(parents=True, exist_ok=True)
    source = input_path(args.input, expected_type)
    if args.command == "compress":
        image = read_image(source)
        if args.target_bpp is not None:
            result = codec.search(image, args.target_bpp)
            blob = result.pop("blob")
            print(result)
        else:
            blob, elapsed = codec.encode(image, args.quality, args.delta)
            print({"encode_seconds": elapsed,
                   "file_bpp": len(blob) * 8 / (image.shape[-2] * image.shape[-1])})
        target.write_bytes(blob)
    else:
        image, elapsed = codec.decode(source.read_bytes())
        save_image(image, target)
        print({"decode_seconds": elapsed})
