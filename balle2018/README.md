# Ballé 2018 scale-hyperprior baseline

Scratch-trained reproduction of the scale-only hyperprior from
[Variational image compression with a scale hyperprior](https://arxiv.org/abs/1802.01436).
Transforms, GDN/IGDN, entropy bottleneck, and rANS primitives come from
[CompressAI](https://github.com/InterDigitalInc/CompressAI) 1.2.8 (BSD-3-Clause).
The implementation uses a zero-mean Gaussian conditional and a factorized
hyperlatent prior. Pretrained weights are not downloaded.

## Setup and paths

Run commands from the repository root using Python 3.12. The original pilots used
Python 3.11; migration checks on Python 3.12 preserve their decoded outputs and
metrics. CPU inference/coding and Apple MPS training have been checked. CUDA is
selected automatically when available; install a CUDA build of PyTorch 2.6 and
torchvision 0.21 for that device.

```text
python -m pip install -r requirements.txt
```

For the pilot's exact core package versions, and optional tests/notebooks:

```text
python -m pip install -r balle2018/requirements-repro.txt
python -m pip install -r balle2018/requirements-dev.txt
```

Input paths are repository-relative, or absolute for external datasets. All CLI
output paths must be inside `balle2018/`. The local directories `data/`, `runs/`,
`results/`, and `.cache/` inside this method are Git-ignored. Checkpoints and data
are local artifacts, not part of a fresh GitHub checkout.

## Data

Download the official 800-image DIV2K HR training set and the 24 original Kodak
images, then create the seed-42 split and fixed validation crops:

```text
python -m balle2018.prepare_data --download
```

This creates 760 training / 40 held-out validation originals, with 512x512 center
crops for frequent validation. Training uses random 256x256 crops and horizontal
flips without resizing. RGB values are in [0,1], with no ImageNet normalization.
Pixel-identical files are excluded before splitting. Dataset source hashes and
crop provenance are recorded under `balle2018/data/`.

To use an existing dataset instead:

```text
python -m balle2018.prepare_data --root path/to/train-images --name dataset-release --output balle2018/data/manifests/custom
python -m balle2018.prepare_data --root path/to/test-images --name test --evaluation --output balle2018/data/manifests/custom
```

Update the configuration's manifest paths. JSON arrays and the team's one-path-
per-line text manifests are supported; paths inside either are manifest-relative.
The original pilot manifests are retained locally byte-for-byte so checkpoint
resume can verify their SHA256s. The variable-rate pilot references the same
original photographs through local relative dataset links.

## Train from scratch

The objective is `R_y + R_z + lambda * 255^2 * RGB_MSE`, with both rates in bpp
per image pixel. Separate Adam optimizers update ordinary weights and entropy-
bottleneck quantiles. Additive uniform noise relaxes quantization during training;
evaluation rounds. Defaults use N=128, M=192, batch 2, 256x256 crops.

```text
python -m balle2018.train --config balle2018/configs/default.json
```

A time-bounded pilot, using a new output directory:

```text
python -m balle2018.train --config balle2018/configs/div2k-pilot.json --output balle2018/runs/new-pilot --max-seconds 5400
```

`--device auto` chooses CUDA, MPS, then CPU. The unsupported MPS Gaussian-CDF
`erfc` operation is evaluated on CPU with gradients preserved; neural transforms
train on the GPU. `--steps`, `--batch-size`, `--lambda`, `--train-manifest`, and
`--val-manifest` override their configuration values. Use separate output paths
for independently trained lambda values.

Resume the local pilot into a new run, preserving checkpoints used by saved files:

```text
python -m balle2018.train --config balle2018/runs/div2k-pilot/config.json --resume balle2018/runs/div2k-pilot/latest.pt --steps 50000 --output balle2018/runs/div2k-continued
```

Checkpoints contain model weights, both optimizers, scheduler, RNG state, step,
phase, configuration, and manifest hashes. Paths may be relocated on resume, but
the manifest hashes and experiment settings must match. SIGINT/SIGTERM request
a clean stop after the current update. `--max-seconds` includes validation time
and is checked between updates; a final validation/checkpoint may slightly exceed
the budget. `best.pt` selects by validation RD loss; `latest.pt` is for continuation.

## Compress and decompress

```text
python -m balle2018 compress --checkpoint balle2018/runs/div2k-pilot/best.pt --input balle2018/data/kodak/kodim01.png --output balle2018/results/example.lic
python -m balle2018 decompress --checkpoint balle2018/runs/div2k-pilot/best.pt --input balle2018/results/example.lic --output balle2018/results/example.png
```

File coding runs on CPU/rANS. The custom `.lic` container includes original
dimensions, control settings, stream lengths, and the checkpoint SHA256. Use the
same checkpoint file for both commands. Retain a stable selected checkpoint with
compressed artifacts. This floating-point codec is not an integer cross-platform
coding standard; the pilot files were checked with four CPU threads on macOS ARM.

## Evaluation and reports

```text
python -m balle2018.evaluate --checkpoints balle2018/runs/div2k-pilot/best.pt --manifest balle2018/data/manifests/div2k-pilot/kodak.json --output balle2018/results/new-evaluation --save-recon --save-bitstreams --timing-repeats 3
python -m balle2018.evaluate --verify-files --output balle2018/results/new-evaluation
python -m balle2018.summarize_results --evaluation balle2018/results/new-evaluation --training-run balle2018/runs/div2k-pilot
```

Outputs include `per_image.csv`, `summary.csv`, `summary.json`, optional PNGs and
`.lic` files, plots, and `REPORT.md`. Every evaluation compares decoded floats with
deterministic model inference. `--verify-files` independently checks saved files
and their recorded distortion. Reports are generated by the general summarizer;
development pilot launchers and notebook-checking scripts are not required.

### Metric conventions

- Full-image **RGB** PSNR/MS-SSIM, no edge crop. Distortion uses decoded float
  outputs clamped to [0,1], before eight-bit PNG rounding.
- Total-file bpp includes the header and both streams, divided by original HxW.
  Payload and likelihood-estimated rates are reported separately.
- Images are edge-padded at bottom/right to multiples of 64; padded symbols count
  in rate, and distortion is cropped to the original dimensions.
- Dataset scores average per image. Pixel-weighted bpp is also recorded.
- MS-SSIM uses five scales and an 11x11 window; it is null when min(H,W)<=160.
  Exact zero MSE has null PSNR, representing mathematical infinity.
- Timings exclude model setup, warm-up, and disk I/O; median repeat per image,
  then dataset mean. Parameter bytes exclude optimizer state.

This differs from the cropped-luma Ballé 2017/ELIC table in the root README.

## Completed local pilot

90.06 minutes on M4/MPS, 24,874 updates; validation selected step 24,750.
Kodak averages over 24 images:

| Total bpp | Payload bpp | RGB PSNR | RGB MS-SSIM |
|---:|---:|---:|---:|
| 0.413879 | 0.408101 | 28.8736 dB | 0.953141 |

All 24 saved files exactly matched deterministic inference. The local artifacts
are in `balle2018/runs/div2k-pilot/` and `balle2018/results/div2k-pilot/`.
This is one short-training operating point, not a complete RD curve or a converged
paper reproduction. Source checkpoints retain their original hashes after migration.

## Tests and notebook

```text
python -m pytest balle2018/tests -q -p no:cacheprovider
python -m jupyter lab balle2018/notebooks
```

`notebooks/baseline_analysis.ipynb` reads method-local artifacts and opens without
them in a fresh checkout. Its committed outputs are cleared. Generated reports
and figures belong in `balle2018/results/`; caches and test fixtures stay local.
