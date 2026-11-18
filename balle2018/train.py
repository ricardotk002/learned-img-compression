"""Scratch training with two optimizers, validation, timed phases, and full resume."""

import argparse
import json
import math
import random
import signal
import time

import torch

from .data import CropDataset, pad_image, read_image, read_manifest
from .losses import rate_distortion
from .model import build_model
from .runtime import (
    choose_device, configure_cache, environment, input_path, load_checkpoint,
    method_name, output_path, restore_rng, rng_state, save_checkpoint, seed_all, sha256, write_json,
)


def configure_optimizers(model, config):
    net, aux = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (aux if name.endswith(".quantiles") else net).append(parameter)
    return (torch.optim.Adam(net, lr=config["learning_rate"]),
            torch.optim.Adam(aux, lr=config["aux_learning_rate"]))


def training_step(model, batch, quality, delta, lmbda, optimizer, aux_optimizer, clip):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    aux_optimizer.zero_grad(set_to_none=True)
    losses = rate_distortion(model(batch, quality, delta), batch, lmbda)
    if not torch.isfinite(losses["loss"]):
        raise FloatingPointError("Non-finite rate-distortion loss")
    losses["loss"].backward()
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
    optimizer.step()
    aux_optimizer.zero_grad(set_to_none=True)
    aux_loss = model.aux_loss()
    if not torch.isfinite(aux_loss):
        raise FloatingPointError("Non-finite entropy-bottleneck auxiliary loss")
    aux_loss.backward()
    aux_optimizer.step()
    return {**{key: value.detach().item() for key, value in losses.items()},
            "aux_loss": aux_loss.item(), "gradient_norm": norm.item()}


@torch.no_grad()
def validate(model, files, lambdas, deltas, device):
    model.eval()
    records = []
    for quality, lmbda in enumerate(lambdas):
        for delta in deltas:
            totals = {key: 0.0 for key in ("loss", "mse", "bpp", "y_bpp", "z_bpp")}
            for path in files:
                x = read_image(path).unsqueeze(0).to(device)
                height, width = x.shape[-2:]
                result = model(pad_image(x), quality, delta)
                result["x_hat"] = result["x_hat"][..., :height, :width]
                values = rate_distortion(result, x, lmbda, num_pixels=height * width)
                for key in totals:
                    totals[key] += values[key].item() / len(files)
            records.append({"quality": quality, "lambda": lmbda, "delta": delta, **totals})
    return sum(row["loss"] for row in records) / len(records), records


def check_config(config):
    required = {"model", "lambdas", "train_manifest", "val_manifest", "output", "seed", "device",
                "num_threads", "crop_size", "batch_size", "steps", "learning_rate",
                "aux_learning_rate", "clip_max_norm", "log_every", "validate_every", "save_every",
                "lr_patience", "lr_factor", "mixed_start_step", "delta_range", "validation_deltas"}
    optional = {"mixed_start_seconds"}
    if required - set(config) or set(config) - required - optional:
        raise ValueError(f"Config keys differ: missing={required - set(config)}, "
                         f"unknown={set(config) - required - optional}")
    if config["model"]["type"] not in {"baseline", "variable"}:
        raise ValueError("Unknown model type")
    if len(config["lambdas"]) != config["model"].get("levels", 1):
        raise ValueError("One lambda is required per quality level")
    if not config["model"].get("conditioning", True) and config["model"]["levels"] != 1:
        raise ValueError("Without conditioning there must be only one quality level")
    if any(not math.isfinite(value) or value <= 0 for value in config["lambdas"]):
        raise ValueError("lambdas must be finite and positive")
    if config["lambdas"] != sorted(set(config["lambdas"])):
        raise ValueError("lambdas must be unique, ordered from low to high quality")
    for key in ("N", "M", "levels"):
        if not isinstance(config["model"][key], int) or config["model"][key] < 1:
            raise ValueError(f"model.{key} must be a positive integer")
    if config["crop_size"] < 64 or config["crop_size"] % 64:
        raise ValueError("crop_size must be a positive multiple of 64")
    for key in ("steps", "batch_size", "log_every", "validate_every", "save_every", "num_threads"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "aux_learning_rate", "clip_max_norm"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not 0 < config["lr_factor"] < 1 or config["lr_patience"] < 0:
        raise ValueError("Invalid learning-rate scheduler settings")
    bounds = config["delta_range"]
    if len(bounds) != 2 or not 0.01 <= bounds[0] <= bounds[1] <= 100:
        raise ValueError("Invalid delta_range")
    if not config["validation_deltas"] or any(
        not math.isfinite(delta) or not 0.01 <= delta <= 100 for delta in config["validation_deltas"]
    ):
        raise ValueError("Invalid validation_deltas")
    start = config["mixed_start_step"]
    if start is not None and (not isinstance(start, int) or start < 0):
        raise ValueError("mixed_start_step must be null or a nonnegative integer")
    seconds = config.get("mixed_start_seconds")
    if seconds is not None:
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("mixed_start_seconds must be finite and nonnegative")
        if start is not None:
            raise ValueError("Use either mixed_start_step or mixed_start_seconds")
    if config["model"]["type"] == "baseline" and (
        start is not None or seconds is not None or bounds != [1, 1]
    ):
        raise ValueError("Baseline training fixes delta=1")


def phase_at(config, step, elapsed_seconds=0):
    seconds = config.get("mixed_start_seconds")
    if seconds is not None:
        return "mixed" if elapsed_seconds >= seconds else "fixed"
    start = config["mixed_start_step"]
    return "mixed" if start is not None and step > start else "fixed"


def run(config, resume=None, max_seconds=None, on_validation=None):
    check_config(config)
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be finite and positive")
    kind = config["model"]["type"]
    configure_cache(kind)
    torch.set_num_threads(config["num_threads"])
    seed_all(config["seed"])
    device = choose_device(config["device"])
    train_manifest = input_path(config["train_manifest"], kind)
    val_manifest = input_path(config["val_manifest"], kind)
    dataset = CropDataset(train_manifest, config["crop_size"])
    val_files = read_manifest(val_manifest)
    if set(dataset.files) & set(val_files):
        raise ValueError("Training and validation manifests overlap")
    identities = {"train_manifest": sha256(train_manifest), "val_manifest": sha256(val_manifest)}
    model = build_model(config["model"]).to(device)
    optimizer, aux_optimizer = configure_optimizers(model, config)

    def scheduler_factory():
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=config["lr_factor"], patience=config["lr_patience"]
        )

    scheduler = scheduler_factory()
    output = input_path(config["output"], kind)
    step, best, phase = 0, math.inf, phase_at(config, 1)
    elapsed_before_resume, transitions = 0.0, []
    quality_steps = [0] * model.levels
    phase_steps = {"fixed": 0, "mixed": 0}
    if resume:
        checkpoint = load_checkpoint(input_path(resume, kind))
        # Dataset paths may move; their byte-identical manifests remain mandatory.
        mutable = {"steps", "output", "device", "num_threads", "log_every", "save_every",
                   "train_manifest", "val_manifest"}
        for key in config.keys() - mutable:
            if config[key] != checkpoint["config"].get(key):
                raise ValueError(f"Resume config changed {key}; start a new experiment instead")
        if config.get("mixed_start_seconds") != checkpoint["config"].get("mixed_start_seconds"):
            raise ValueError("Resume cannot change the timed phase schedule")
        if identities != checkpoint["manifest_hashes"]:
            raise ValueError("Dataset manifests changed since the checkpoint")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        step, best, phase = checkpoint["step"], checkpoint["best"], checkpoint["phase"]
        elapsed_before_resume = checkpoint.get("training_elapsed_seconds", 0.0)
        transitions = checkpoint.get("phase_transitions", [])
        quality_steps = checkpoint.get("quality_updates", [0] * model.levels)
        phase_steps = checkpoint.get("phase_updates", {"fixed": 0, "mixed": 0})
        restore_rng(checkpoint["rng"])
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Nonempty output directory: {output}; use --resume or a new output")
    if step >= config["steps"]:
        raise ValueError("--steps must exceed the resumed checkpoint step")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_json(output / "environment.json", {**environment(), "device": str(device),
               "manifest_hashes": identities, "sampling": "uniform with replacement"})
    print(f"Device: {device}; architecture: {config['model']}; "
          f"train={len(dataset)}, validation={len(val_files)}", flush=True)
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True
        print("Stopping after this step and saving latest.pt...", flush=True)

    previous_handler = signal.signal(signal.SIGINT, request_stop)
    previous_term_handler = signal.signal(signal.SIGTERM, request_stop)
    started, start_step = time.perf_counter(), step

    def total_elapsed():
        return elapsed_before_resume + time.perf_counter() - started

    def state():
        return {"format_version": 1, "config": config, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "aux_optimizer": aux_optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "step": step, "best": best, "phase": phase,
                "rng": rng_state(), "manifest_hashes": identities,
                "training_elapsed_seconds": total_elapsed(), "phase_transitions": transitions,
                "quality_updates": quality_steps, "phase_updates": phase_steps}

    try:
        while step < config["steps"]:
            next_step = step + 1
            next_phase = phase_at(config, next_step, total_elapsed())
            if next_phase != phase:
                save_checkpoint(output / "stage1.pt", state())
                transitions.append({"step": next_step, "elapsed_seconds": total_elapsed(),
                                    "from": phase, "to": next_phase})
                phase, best = next_phase, math.inf
                for group in optimizer.param_groups:
                    group["lr"] = config["learning_rate"]
                scheduler = scheduler_factory()
                print(f"Step {next_step}: mixed-step training; reset LR and validation best", flush=True)
            quality = random.randrange(model.levels)
            delta = 1.0
            if phase == "mixed":
                lo, hi = config["delta_range"]
                delta = 2 ** random.uniform(math.log2(lo), math.log2(hi))
            batch = torch.stack([dataset[random.randrange(len(dataset))]
                                 for _ in range(config["batch_size"])]).to(device)
            metrics = training_step(model, batch, quality, delta, config["lambdas"][quality],
                                    optimizer, aux_optimizer, config["clip_max_norm"])
            step = next_step
            quality_steps[quality] += 1
            phase_steps[phase] += 1
            if max_seconds is not None and time.perf_counter() - started >= max_seconds:
                stop = True
            if step % config["log_every"] == 0 or step == 1 or stop:
                record = {"step": step, "phase": phase, "quality": quality, "delta": delta,
                          "lambda": config["lambdas"][quality], "device": str(device),
                          "lr": optimizer.param_groups[0]["lr"], **metrics,
                          "quality_updates": quality_steps.copy(),
                          "seconds_per_step": (time.perf_counter() - started) / (step - start_step)}
                with (output / "train.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            if step % config["validate_every"] == 0 or step == config["steps"] or stop:
                deltas = config["validation_deltas"] if phase == "mixed" else [1.0]
                score, records = validate(model, val_files, config["lambdas"], deltas, device)
                scheduler.step(score)
                record = {"step": step, "phase": phase, "loss": score, "operating_points": records}
                with (output / "validation.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(record) + "\n")
                print(f"Validation step {step} ({phase}): {score:.6f}", flush=True)
                if score < best:
                    best = score
                    save_checkpoint(output / "best.pt", state())
                    save_checkpoint(output / f"best-{phase}.pt", state())
                if on_validation is not None:
                    on_validation(model, step, records, output)
            if max_seconds is not None and time.perf_counter() - started >= max_seconds:
                stop = True
            if step % config["save_every"] == 0 or step == config["steps"] or stop:
                save_checkpoint(output / "latest.pt", state())
            if stop:
                break
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        signal.signal(signal.SIGTERM, previous_term_handler)
    write_json(output / "session.json", {"start_step": start_step, "end_step": step,
               "elapsed_seconds": time.perf_counter() - started, "max_seconds": max_seconds,
               "stopped_early": step < config["steps"], "device": str(device),
               "total_training_elapsed_seconds": total_elapsed(), "phase_transitions": transitions,
               "quality_updates": quality_steps, "phase_updates": phase_steps})
    return output / "latest.pt"


def main(expected_type="baseline"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=f"{method_name(expected_type)}/configs/default.json")
    parser.add_argument("--resume")
    parser.add_argument("--output")
    parser.add_argument("--train-manifest")
    parser.add_argument("--val-manifest")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--lambda", dest="lmbda", type=float)
    args = parser.parse_args()
    config = json.loads(input_path(args.config, expected_type).read_text(encoding="utf-8"))
    if config["model"]["type"] != expected_type:
        parser.error(f"This entry point requires model.type={expected_type}")
    for key in ("output", "train_manifest", "val_manifest", "device", "steps", "batch_size"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    config["output"] = str(output_path(config["output"], expected_type, "runs/training"))
    if args.lmbda is not None:
        if expected_type != "baseline":
            parser.error("--lambda overrides only fixed-rate baseline runs")
        config["lambdas"] = [args.lmbda]
    run(config, args.resume, max_seconds=args.max_seconds)


if __name__ == "__main__":
    main()
