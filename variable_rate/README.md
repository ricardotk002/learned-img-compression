# Conditional variable-rate hyperprior

One scale-hyperprior checkpoint supports discrete quality levels and continuous
quantization-step control, inspired by [Choi et al., ICCV 2019](https://arxiv.org/abs/1909.04802).
It reuses `balle2018`'s transforms, coding primitives, training engine, and RGB
evaluation protocol. Learned per-quality channel scales/biases condition each
convolution in the analysis/synthesis and hyperprior transforms.

This is a mechanism adaptation: it retains the scale-only Gaussian prior,
factorized hyperlatents, and additive-noise training, rather than reproducing the
paper's full autoregressive architecture and universal quantization.

## Setup and local artifacts

Run from the repository root with Python 3.12:

```text
python -m pip install -r requirements.txt
```

Exact pilot package versions and optional test/notebook dependencies:

```text
python -m pip install -r variable_rate/requirements-repro.txt
python -m pip install -r balle2018/requirements-dev.txt
```

The pilots originally used Python 3.11; Python 3.12 checks preserve the trained
models' file decoding and scores. All CLI outputs stay inside `variable_rate/`:
`data/` contains manifests/prepared inputs, `runs/` checkpoints/logs, `results/`
evaluation files/figures/reports, and `.cache/` temporary library/test files.
These artifacts are Git-ignored. Configurations, tests, source, and the analysis
notebook are tracked. No background pilot launcher or smoke runner is needed.

## Dataset preparation

Use the same DIV2K originals and Kodak images as the baseline. On a fresh checkout,
download them with `python -m balle2018.prepare_data --download`, then create
method-local manifests without downloading a second copy:

```text
python -m variable_rate.prepare_data --root balle2018/data/div2k/DIV2K_train_HR --name div2k-train-hr
python -m variable_rate.prepare_data --root balle2018/data/kodak --name kodak --evaluation
```

Both methods use seed 42, 760 training / 40 held-out validation originals, and
fixed 512x512 validation center crops. Training samples random 256x256 RGB crops
without resizing. Dataset paths can point to any existing image directory;
generated manifests/crops remain method-local. JSON and team-style text manifests
are both supported. Existing migrated pilot manifests remain byte-identical to
their originals; ignored relative links under `variable_rate/data/` reference
the raw photographs in `balle2018/data/` for local resume compatibility.

## Scratch training

Four qualities use lambdas `[0.0018, 0.0035, 0.0067, 0.013]` in
`R_y + R_z + lambda * 255^2 * RGB_MSE` convention. One quality is sampled uniformly
per minibatch. Higher quality indices generally raise rate and improve quality.
Choi uses `D + lambda R`, so its numerical lambdas are not interchangeable.

```text
python -m variable_rate.train --config variable_rate/configs/default.json
```

The general configuration has a step-based switch: 100k updates with delta=1,
then 100k with log-uniform delta in [0.5,2]. For the short timed experiment:

```text
python -m variable_rate.train --config variable_rate/configs/div2k-pilot.json --output variable_rate/runs/new-pilot --max-seconds 5400
```

This initializes randomly, trains for 60 minutes with delta=1, then continues the
same weights and optimizers for 30 minutes with mixed deltas. Both the phase
schedule and time budget include validation time. The final validation/save can
slightly exceed the budget. Cumulative phase time is checkpointed for resume.
`mixed_start_step` and `mixed_start_seconds` are mutually exclusive. CPU/MPS/CUDA
device selection and optimizer details are described in `balle2018/README.md`.

Correct variable quantization/coding uses:

```text
k = round(y / delta)
p(k | z_hat) = unit-bin Gaussian masses with scale sigma(z_hat) / delta
y_hat = delta * k
```

Hyperlatent quantization remains at unit step. The file header stores quality and
delta for decoding. Smaller delta generally increases rate. Finite files have
discrete rates, so exact continuous target rates are not guaranteed.

Resume into a new run, retaining the selected checkpoint used by existing files:

```text
python -m variable_rate.train --config variable_rate/runs/div2k-pilot/config.json --resume variable_rate/runs/div2k-pilot/latest.pt --steps 50000 --output variable_rate/runs/div2k-continued
```

Checkpoint selection averages validation RD losses across qualities and, during
mixed training, the configured validation deltas. Best score and LR scheduler
reset at the phase transition. `best-fixed.pt` and `best-mixed.pt` identify each
phase's best; `stage1.pt` stores the boundary state, and `latest.pt` resumes.
Fixed- and mixed-phase scores average different operating points.

Two additional scratch-trained ablation configurations are provided:

```text
python -m variable_rate.train --config variable_rate/configs/conditioning-only.json
python -m variable_rate.train --config variable_rate/configs/quantization-only.json
```

These ablations have not been run as real-data experiments. The quantization-only
model has one lambda and no quality-conditioning modules.

## Compression and evaluation

```text
python -m variable_rate compress --checkpoint variable_rate/runs/div2k-pilot/best-mixed.pt --input balle2018/data/kodak/kodim01.png --output variable_rate/results/example.lic --quality 3 --delta 0.75
python -m variable_rate decompress --checkpoint variable_rate/runs/div2k-pilot/best-mixed.pt --input variable_rate/results/example.lic --output variable_rate/results/example.png
```

Coding runs on CPU/rANS. Use the same unchanged checkpoint file for decoding;
the container verifies its SHA256. The container and float precision conventions
are shared with `balle2018`.

Evaluate the 20 predeclared pilot controls, then verify and summarize:

```text
python -m variable_rate.evaluate --checkpoints variable_rate/runs/div2k-pilot/best-mixed.pt --manifest variable_rate/data/manifests/div2k-pilot/kodak.json --output variable_rate/results/new-evaluation --deltas 0.5 0.75 1 1.5 2 --save-recon --save-bitstreams --timing-repeats 3
python -m variable_rate.evaluate --verify-files --output variable_rate/results/new-evaluation
python -m variable_rate.summarize_results --evaluation variable_rate/results/new-evaluation --training-run variable_rate/runs/div2k-pilot --compare balle2018/results/div2k-pilot
```

The shared protocol uses **full-image RGB** PSNR/MS-SSIM on clamped decoded float
outputs, with no edge crop. Whole-file bpp includes both streams and headers per
original image pixel. See the baseline README for padding, timing, and aggregation.
This differs from the root README's cropped-luma Ballé 2017/ELIC table.

Reports include raw operating points and their nondominated RD frontier, control
sweeps, monotonicity diagnostics, and coarse grid target-rate errors. `--compare`
requires the same evaluation manifest hash. Optional `--anchor baseline` computes
PSNR BD-rate only if both curves have at least four nondominated points; it cannot
be used with the single current baseline point.

Refined actual-file-rate search and diagnostics are also available:

```text
python -m variable_rate compress --checkpoint variable_rate/runs/div2k-pilot/best-mixed.pt --input balle2018/data/kodak/kodim01.png --output variable_rate/results/target.lic --target-bpp 0.3
python -m variable_rate.rate_control diagnose --per-image variable_rate/results/div2k-pilot/per_image.csv --output variable_rate/results/div2k-pilot/rate-control.json
python -m variable_rate.rate_control targets --checkpoint variable_rate/runs/div2k-pilot/best-mixed.pt --manifest variable_rate/data/manifests/div2k-pilot/kodak.json --bpps 0.25 0.4 --output variable_rate/results/targets
```

Refined search can take substantially longer than a single encode. It uses bytes
to choose the nearest tested candidate and reports attainable range and error.

## Completed local pilot

90.64 minutes on M4/MPS, 22,522 updates (15,563 fixed-step / 6,959 mixed-step).
Mixed-stage validation selected step 20,000. Kodak means at delta=1:

| Quality | Total bpp | RGB PSNR | RGB MS-SSIM |
|---:|---:|---:|---:|
| 0 | 0.23584 | 26.739 dB | 0.92576 |
| 1 | 0.30250 | 27.572 dB | 0.93958 |
| 2 | 0.35861 | 27.933 dB | 0.94711 |
| 3 | 0.39694 | 28.172 dB | 0.95074 |

The 20 controls span 0.11852–0.66696 bpp and 25.20–28.71 dB. All 480 files exactly
matched deterministic inference. There were zero rate reversals in 384 delta
comparisons and 360 quality comparisons. These results demonstrate useful rate
control, while the fixed-rate pilot is more PSNR-efficient near 0.4 bpp:
28.874 dB at 0.41388 bpp versus 28.172 dB at 0.39694 bpp. These are nearby rates,
not an exact matched-rate BD-rate comparison.

Equal wall time is shared across four objectives; this does not establish an
inherent variable-rate penalty or a converged reproduction. Artifacts remain in
`variable_rate/runs/div2k-pilot/` and `variable_rate/results/div2k-pilot/` locally.

## Tests and notebook

```text
python -m pytest variable_rate/tests -q -p no:cacheprovider
python -m jupyter lab variable_rate/notebooks
```

`notebooks/variable_rate_analysis.ipynb` reads both methods' local results, plots
RD/control sweeps, and examines target-rate errors. It opens without artifacts in
a fresh checkout. Committed notebook outputs are cleared; generated figures and
reports stay under `variable_rate/results/`.
