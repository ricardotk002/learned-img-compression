# Balle 2017 baseline

Implementation based on [End-to-end Optimized Image Compression](https://arxiv.org/abs/1611.01704).
Includes training, file compression/decompression, and evaluation.

## Install

Use Python 3.12. Run commands from the repository root in your terminal on
Windows, Linux, or macOS. Use `python3` instead of `python` if required by
your installation:

```text
python -m pip install -r requirements.txt
```

For NVIDIA GPU training, install a CUDA-enabled PyTorch build using the
[PyTorch installation instructions](https://pytorch.org/get-started/locally/).
Run tests:

```text
python -m unittest discover -s balle2017/tests -v
```

## Prepare data

Download full-resolution Imagenette and Kodak and generate separate manifests:

```text
python -m balle2017.prepare_data --download --saturation-fraction 0.05
```

Outputs are stored in `data/`, including `train.txt`, `val.txt`, `kodak.txt`,
and dataset provenance. For another dataset, provide UTF-8 manifests containing
one image path per line; relative paths resolve from the manifest directory.
Training images must support 256x256 crops after downsampling below 0.75.

## Train

```text
python -m balle2017.train --train-manifest data/train.txt --val-manifest data/val.txt --output runs/balle2017/gray-lambda512 --mode gray --lambda 512 --batch-size 8 --steps 10000 --validate-every 250 --saturation-fraction 0.05 --device cuda
```

Use `--device cpu` for CPU training. Grayscale defaults to 128 channels; RGB
uses 192. Train a separate model for each lambda. Validation uses fixed 256x256
center crops; `--validation-crop 0` evaluates full images.

The output directory contains `status.json`, `run.json`, `metrics.jsonl`,
`last.pt`, and `best.pt`. Training progress is printed in the terminal;
`status.json` records progress and `metrics.jsonl` records validation results.

Resume from the latest checkpoint, targeting 20,000 total steps:

```text
python -m balle2017.train --train-manifest data/train.txt --val-manifest data/val.txt --output runs/balle2017/gray-lambda512 --mode gray --lambda 512 --batch-size 8 --steps 20000 --validate-every 250 --saturation-fraction 0.05 --device cuda --resume runs/balle2017/gray-lambda512/last.pt
```

## Compress and decompress

```text
python -m balle2017 compress --checkpoint runs/balle2017/gray-lambda512/best.pt --input data/example.png --output runs/example.b17
python -m balle2017 decompress --checkpoint runs/balle2017/gray-lambda512/best.pt --input runs/example.b17 --output runs/reconstruction.png
```

Both commands require the same checkpoint. File coding runs on CPU. The codec
uses an adaptive reference arithmetic coder rather than the original CABAC
implementation; `.b17` is a custom container format.

## Evaluate

```text
python -m balle2017.evaluate --checkpoints runs/balle2017/gray-lambda512/best.pt --test-manifest data/kodak.txt --output runs/balle2017/kodak --jpeg-quality 10 20 30 40 50 70 90 --jpeg2000-rates 8 10 12 16 20 30 40 --matched-classical
```

Outputs include per-image CSV metrics, JSON summaries, reconstructed PNGs, and
compressed files. `--matched-classical` searches for JPEG/JPEG2000 file sizes
near the learned file size and records the remaining rate mismatch.
JPEG2000 arguments are compression ratios, not target bits per pixel.

Evaluation removes eight pixels from each Kodak edge by default. Use
`--crop-border 0` for already-cropped images or another dataset. Metrics use raw
floating reconstructions; PNG previews are clipped and rounded.

Generate plots and a report from evaluation outputs:

```text
python -m balle2017.summarize_results --evaluation runs/balle2017/kodak --training-run runs/balle2017/gray-lambda512
```

The report generator expects one checkpoint evaluated on all 24 Kodak images
with `--matched-classical` enabled.

## Import numeric transform weights

For compatible MATLAB transform files in a local directory:

```text
python -m balle2017.import_weights --directory path/to/weights --lambda 512 --output runs/imported.pt
python -m balle2017.verify_reference --directory path/to/weights
```

Imported transforms do not include a fitted entropy density. Rate evaluation
is disabled until the density is trained. Verification compares the transforms
against independent NumPy calculations using the supplied arrays.
