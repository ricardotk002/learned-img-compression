import copy
import json
import shutil

import pytest
import torch

from balle2018.codec import Codec
from balle2018.data import pad_image
from balle2018.model import build_model
from balle2018.runtime import ROOT, load_checkpoint, save_checkpoint
from balle2018.tests.fixtures import make_data
from balle2018.train import configure_optimizers, phase_at, run, training_step


def config_for(tmp_path):
    make_data(tmp_path / "data")
    config = json.loads((ROOT / "variable_rate/configs/default.json").read_text())
    config.update({"model": {"type": "variable", "N": 8, "M": 12, "levels": 2},
                   "lambdas": [0.0035, 0.013], "device": "cpu", "num_threads": 2,
                   "crop_size": 64, "batch_size": 1, "steps": 4, "mixed_start_step": 2,
                   "log_every": 1, "save_every": 1, "validate_every": 1,
                   "train_manifest": str(tmp_path / "data/train.json"),
                   "val_manifest": str(tmp_path / "data/val.json"),
                   "output": str(tmp_path / "full")})
    return config


def test_resume_and_relocated_manifests_match_uninterrupted_training(tmp_path):
    config = config_for(tmp_path)
    full_path = run(config)
    split = copy.deepcopy(config)
    split.update({"output": str(tmp_path / "split"), "steps": 2})
    split_path = run(split)
    shutil.copytree(tmp_path / "data", tmp_path / "relocated")
    split.update({"steps": 4, "train_manifest": str(tmp_path / "relocated/train.json"),
                  "val_manifest": str(tmp_path / "relocated/val.json")})
    resumed_path = run(split, split_path)
    full, resumed = load_checkpoint(full_path), load_checkpoint(resumed_path)
    assert full["phase"] == resumed["phase"] == "mixed"
    for key, expected in full["model"].items():
        torch.testing.assert_close(resumed["model"][key], expected, rtol=0, atol=0)
    assert full["scheduler"] == resumed["scheduler"]
    assert full["quality_updates"] == resumed["quality_updates"]
    assert (tmp_path / "split/stage1.pt").is_file()


def test_timed_phase_transition_persists_on_resume(tmp_path):
    config = config_for(tmp_path)
    config.update({"steps": 1, "mixed_start_step": None, "mixed_start_seconds": 1e-8})
    assert phase_at(config, 999999, 0) == "fixed"
    assert phase_at(config, 1, 1) == "mixed"
    path = run(config)
    first = load_checkpoint(path)
    assert first["phase"] == "mixed"
    config["steps"] = 2
    run(config, resume=path)
    second = load_checkpoint(path)
    assert second["phase_transitions"] == first["phase_transitions"]
    assert second["training_elapsed_seconds"] >= first["training_elapsed_seconds"]
    assert (path.parent / "best-mixed.pt").is_file()


def test_nonzero_variable_latents_roundtrip_all_controls(tmp_path):
    model = build_model({"type": "variable", "N": 8, "M": 12, "levels": 2})
    optimizer, aux = configure_optimizers(model, {"learning_rate": 1e-4, "aux_learning_rate": 1e-3})
    for quality in range(2):
        training_step(model, torch.rand(1, 3, 64, 64), quality, 0.7, 0.0067, optimizer, aux, 1)
    with torch.no_grad():
        model.g_a[-1].bias.add_(2)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, {"config": {"model": model.model_config,
                                           "lambdas": [0.0035, 0.013]},
                                  "model": model.state_dict(), "step": 2})
    codec = Codec(checkpoint)
    x = torch.rand(1, 3, 67, 91)
    for quality in range(2):
        for delta in (0.5, 0.7, 1.0, 2.0):
            blob, _ = codec.encode(x, quality, delta)
            decoded, _ = codec.decode(blob)
            with torch.no_grad():
                expected = codec.model(pad_image(x), quality, delta)["x_hat"][..., :67, :91].clamp(0, 1)
            torch.testing.assert_close(decoded, expected, rtol=0, atol=1e-6)


def test_target_search_measures_full_file_bytes(tmp_path):
    model = build_model({"type": "variable", "N": 8, "M": 12, "levels": 2})
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, {"config": {"model": model.model_config,
                                           "lambdas": [0.0035, 0.013]},
                                  "model": model.state_dict()})
    codec = Codec(checkpoint)
    image = torch.rand(3, 64, 64)
    blob, _ = codec.encode(image, 1, 1)
    target = len(blob) * 8 / 4096
    result = codec.search(image, target, iterations=1)
    assert result["bpp"] == len(result["blob"]) * 8 / 4096
    assert abs(result["error_bpp"]) < 0.01


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_updates_transform_and_selected_condition():
    model = build_model({"type": "variable", "N": 8, "M": 12, "levels": 2}).to("mps")
    optimizer, aux = configure_optimizers(model, {"learning_rate": 1e-4, "aux_learning_rate": 1e-3})
    before = model.g_a[0].weight.detach().clone()
    condition = model.conditions["g_a_0"].scale.weight.detach().clone()
    result = training_step(model, torch.rand(1, 3, 64, 64, device="mps"), 1, 0.7,
                           0.0067, optimizer, aux, 1)
    assert result["gradient_norm"] > 0
    assert not torch.equal(before, model.g_a[0].weight)
    assert not torch.equal(condition[1], model.conditions["g_a_0"].scale.weight[1])
    torch.testing.assert_close(condition[0], model.conditions["g_a_0"].scale.weight[0])
