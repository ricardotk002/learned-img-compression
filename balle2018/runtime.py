"""Device, checkpoint, and repository-relative path utilities for both methods."""

import hashlib
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def method_name(kind):
    if kind not in {"baseline", "variable"}:
        raise ValueError("Unknown method")
    return "balle2018" if kind == "baseline" else "variable_rate"


def configure_cache(kind="baseline"):
    cache = ROOT / method_name(kind) / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
    os.environ["TORCH_HOME"] = str(cache / "torch")
    return cache


def input_path(value, kind="baseline"):
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def checkpoint_input_path(value, kind, config):
    """Resolve the original pilot's data/... metadata without changing its bytes."""
    path = Path(value)
    if path.is_absolute():
        return path
    legacy_output = "runs/baseline-div2k-pilot" if kind == "baseline" else "runs/variable-div2k-pilot"
    if config.get("output") == legacy_output and path.parts and path.parts[0] == "data":
        return ROOT / method_name(kind) / path
    return ROOT / path


def output_path(value, kind, default):
    path = ROOT / method_name(kind) / default if value is None else input_path(value, kind)
    path = path.resolve()
    if not path.is_relative_to((ROOT / method_name(kind)).resolve()):
        raise ValueError(f"Outputs for this method must be inside {method_name(kind)}/")
    return path


def portable_path(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def choose_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable; use --device cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def environment():
    import compressai

    return {"python": platform.python_version(), "torch": torch.__version__,
            "compressai": compressai.__version__, "platform": platform.platform()}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_checkpoint(path, state):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def load_model(path, device="cpu", coding=False):
    from balle2018.model import build_model

    checkpoint = load_checkpoint(path)
    model = build_model(checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if coding:
        model.update(force=True)
    return model.to(device), checkpoint
