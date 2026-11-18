import json

import numpy as np
import pytest
import torch
from compressai.models import ScaleHyperprior

from balle2018.analysis import bd_rate
from balle2018.bitstream import pack, unpack
from balle2018.codec import Codec
from balle2018.data import pad_image, read_manifest
from balle2018.evaluate import evaluate, verify_files
from balle2018.metrics import image_metrics
from balle2018.model import build_model
from balle2018.prepare_data import prepare
from balle2018.runtime import ROOT, input_path, load_checkpoint, output_path, save_checkpoint
from balle2018.train import check_config, configure_optimizers, run, training_step
from .fixtures import make_data


def save_model(path, model):
    save_checkpoint(path, {"config": {"model": model.model_config, "lambdas": [0.0067]},
                           "model": model.state_dict(), "step": 0})


def test_baseline_matches_reference_forward_and_streams():
    torch.manual_seed(12)
    reference = ScaleHyperprior(16, 24).eval()
    model = build_model({"type": "baseline", "N": 16, "M": 24, "levels": 1}).eval()
    model.load_state_dict(reference.state_dict())
    x = torch.rand(1, 3, 64, 128)
    with torch.no_grad():
        expected, actual = reference(x), model(x)
    torch.testing.assert_close(expected["x_hat"], actual["x_hat"], rtol=0, atol=0)
    for name in ("y", "z"):
        torch.testing.assert_close(expected["likelihoods"][name], actual["likelihoods"][name])
    reference.update(force=True)
    model.update(force=True)
    assert reference.compress(x) == model.compress(x)


def test_nonzero_latent_file_roundtrip(tmp_path):
    model = build_model({"type": "baseline", "N": 8, "M": 12, "levels": 1})
    optimizer, aux = configure_optimizers(model, {"learning_rate": 1e-4, "aux_learning_rate": 1e-3})
    training_step(model, torch.rand(1, 3, 64, 64), 0, 1, 0.0067, optimizer, aux, 1)
    with torch.no_grad():
        model.g_a[-1].bias.add_(2)
    path = tmp_path / "model.pt"
    save_model(path, model)
    codec = Codec(path)
    x = torch.rand(1, 3, 67, 91)
    with torch.no_grad():
        assert (codec.model.g_a(pad_image(x)).round() != 0).any()
        expected = codec.model(pad_image(x))["x_hat"][..., :67, :91].clamp(0, 1)
    blob, _ = codec.encode(x)
    stream = tmp_path / "image.lic"
    stream.write_bytes(blob)
    decoded, _ = Codec(path).decode(stream.read_bytes())
    torch.testing.assert_close(decoded, expected, rtol=0, atol=1e-6)
    codec.checkpoint_hash = "wrong"
    with pytest.raises(ValueError, match="different checkpoint"):
        codec.decode(blob)


def test_container_rejects_truncated_payload():
    blob = pack({"strings": [[b"1234"], [b"5678"]], "shape": [1, 1]}, 64, 64, 0, 1, "x", {})
    with pytest.raises(ValueError, match="length mismatch"):
        unpack(blob[:-1])


def test_actual_evaluation_and_independent_verification(tmp_path):
    make_data(tmp_path / "data")
    model = build_model({"type": "baseline", "N": 8, "M": 12, "levels": 1})
    path = tmp_path / "model.pt"
    save_model(path, model)
    point = evaluate([path], tmp_path / "data/test.json", tmp_path / "eval",
                     save_recon=True, save_bitstreams=True, threads=2)[0]
    assert point["file_bpp"] == pytest.approx(point["payload_bpp"] + point["header_bpp"])
    assert point["payload_bpp"] == pytest.approx(point["y_bpp"] + point["z_bpp"])
    assert 0 <= point["ms_ssim"] <= 1
    assert verify_files(tmp_path / "eval", threads=2)["files"] == 1
    exact = image_metrics(torch.zeros(1, 3, 64, 64), torch.zeros(1, 3, 64, 64))
    assert exact == {"mse": 0, "psnr": None, "ms_ssim": None}


def test_budgeted_training_saves_completed_step(tmp_path):
    make_data(tmp_path / "data")
    config = json.loads((ROOT / "balle2018/configs/default.json").read_text())
    config.update({"model": {"type": "baseline", "N": 8, "M": 12, "levels": 1},
                   "train_manifest": str(tmp_path / "data/train.json"),
                   "val_manifest": str(tmp_path / "data/val.json"),
                   "output": str(tmp_path / "training"), "device": "cpu", "num_threads": 2,
                   "batch_size": 1, "crop_size": 64, "steps": 20})
    validations = []
    path = run(config, max_seconds=1e-8,
               on_validation=lambda model, step, rows, output: validations.append(step))
    assert load_checkpoint(path)["step"] == 1
    assert validations == [1]
    assert (path.parent / "best.pt").is_file()


def test_manifests_deduplicate_and_support_team_text_format(tmp_path):
    make_data(tmp_path / "data")
    (tmp_path / "data/copy.png").write_bytes((tmp_path / "data/image0.png").read_bytes())
    splits = prepare(tmp_path / "data", tmp_path / "manifests", "synthetic", 0.2)
    assert len(splits["train"]) + len(splits["val"]) == 6
    assert not set(splits["train"]) & set(splits["val"])
    text = tmp_path / "data/images.txt"
    text.write_text("# dataset\nimage0.png\nimage1.png\n")
    assert len(read_manifest(text)) == 2


def test_bd_rate_known_savings_without_extrapolation():
    rates = np.array([0.1, 0.2, 0.4, 0.8])
    qualities = [28, 30, 32, 34]
    assert bd_rate(rates, qualities, rates * 0.8, qualities)["bd_rate_percent"] == pytest.approx(-20)
    with pytest.raises(ValueError, match="overlapping"):
        bd_rate(rates, qualities, rates, [35, 36, 37, 38])
    with pytest.raises(ValueError, match="four"):
        bd_rate(rates[:3], qualities[:3], rates, qualities)


def test_configs_and_method_local_output_paths():
    assert input_path("data/train.txt") == ROOT / "data/train.txt"
    for method in ("balle2018", "variable_rate"):
        for path in (ROOT / method / "configs").glob("*.json"):
            check_config(json.loads(path.read_text()))
    assert output_path(None, "baseline", "results/test").is_relative_to(ROOT / "balle2018")
    with pytest.raises(ValueError, match="inside balle2018"):
        output_path("results/wrong", "baseline", "results/test")
