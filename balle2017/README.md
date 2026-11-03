# Ballé 2017: End-to-end Optimized Image Compression

Pranjal's baseline for ECE-GY 6123. Paper: Ballé, Laparra, Simoncelli, ICLR 2017,
[arXiv:1611.01704](https://arxiv.org/abs/1611.01704).

**Status: working initial reconstruction, not a completed historical reproduction.**
The network and spline training follow the 2017 method. Original CABAC parity,
training-set identity, and published rate-distortion parity remain unverified.
Read [FIDELITY.md](FIDELITY.md) before interpreting results. An initial
natural-image Imagenette training experiment was launched on 2026-11-02;
it is not a reproduction of the paper's training subset or published results.

## Setup

From the repository root, with Python 3.12:

```powershell
python -m pip install --force-reinstall torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 --no-deps
python -m pip install -r requirements.txt
python -m unittest discover -s balle2017/tests -v
```

Use the system-wide Python installation; the project owner explicitly prefers
this over a project-specific virtual environment. Do not create a `.venv`.
The development machine has an RTX 3070 Ti Laptop GPU (8 GB). The CUDA build
selected for this setup is PyTorch 2.11.0 with CUDA 12.8 from the official index.
For other machines, consult the [official selector](https://pytorch.org/get-started/locally/).

Verify the actual baseline's forward/backward/optimizer operations on the GPU:

```powershell
python -m balle2017.check_gpu
```

This uses synthetic input and saves `runs/gpu-check.json`; it is not a full
training run. No environment activation is required.

## Data

For the approved preliminary run, full-resolution Imagenette v2 and all 24 Kodak
PNGs are downloaded under ignored `data/`. Reproduce the download/preparation:

```powershell
python -m balle2017.prepare_data --download --saturation-fraction 0.05
```

The prepared split has **4,726 training / 256 validation / 24 test images**.
The requested 6,507 training images cannot be reached from Imagenette's supplied
training split after filtering; all eligible images are used. The class-restricted
ImageNet subset is an explicit deviation approved for this first run.
`data/dataset_provenance.json` records source URLs, archive/image hashes, rejected
images and deterministic selection. Training/validation/test are content-disjoint.

Obtain ImageNet through your institutional access. We do not redistribute it.
Create UTF-8 text manifests, one image path per line; relative paths resolve
against the manifest directory. Keep training, validation, and testing disjoint.
The original 6,507-image subset and exact filtering settings are unavailable.
Keep the accepted/rejected image list in the generated `run.json`.

Preprocessing: reject saturated/small images, add pixel quantization noise,
randomly downsample by less than 0.75, crop 256x256, then convert to grayscale
for grayscale training. The default saturation threshold and resizing algorithm
are explicit reconstruction choices, not reported historical settings.

Kodak: supply the 24 original images as a test manifest. Evaluation defaults
to removing eight pixels from each edge, matching the authors' project page.
Use `--crop-border 0` for images already cropped or another dataset.

## Train one operating point

```powershell
python -m balle2017.train --train-manifest data/train.txt --val-manifest data/val.txt --output runs/balle2017/gray-lambda512 --mode gray --lambda 512 --batch-size 8 --device cuda
```

Each lambda needs a separate run. Released grayscale weights cover powers of
two from 32 through 4096. Start with grayscale (128 channels); RGB defaults to
192 channels. Use `--channels 256` only when explicitly selecting a high-rate
variant. The original per-operating-point channel choices need verification.

`last.pt` supports resuming optimizer state with `--resume PATH` and the same
mode/lambda/output arguments. DataLoader order after a resume is not bit-exact.
`best.pt` is selected by rounded validation rate-distortion loss. The training
log separates rate from distortion. All input pixels are scaled to [0,1].
The objective is bpp + lambda * summed squared channel errors per pixel;
RGB distortion is three times channel-averaged MSE. Historical lambda scaling
must be checked before comparing RGB operating points numerically.

The support of each spline is initially [-30,30] with 0.1 knot spacing. Training
expands support before overflow (two-unit margin, at least 25% growth), preserves
old knots, adds small positive tail knots, renormalizes and rebinds the plain SGD
optimizer. A periodic million-step check is retained. This conservative, grow-only
heuristic is our reconstruction of an unspecified historical detail; it does not
claim original-heuristic parity. Validation/inference do not silently change it.

## First natural-image run

```powershell
.\balle2017\start_training.ps1
# If interrupted after a checkpoint exists:
.\balle2017\start_training.ps1 -Resume
```

The launcher uses system-wide Python and a hidden background process. It targets
10,000 steps, grayscale/128 channels, lambda 512, batch 8, Adam 1e-4, spline SGD
0.01, seed 2017, validation/checkpointing every 250 steps on deterministic 256x256
center crops of held-out images. Kodak is never used for checkpoint selection.
Logs, PID, `status.json`, `run.json`, `last.pt` and `best.pt` live in
`runs/balle2017/imagenette-gray-lambda512-20261102/`.
See [FIRST_RUN.md](FIRST_RUN.md) for the decisions and commands.

## Encode and decode

```powershell
python -m balle2017 compress --checkpoint runs/balle2017/gray-lambda512/best.pt --input data/example.png --output runs/example.b17
python -m balle2017 decompress --checkpoint runs/balle2017/gray-lambda512/best.pt --input runs/example.b17 --output runs/reconstruction.png
```

The stream stores dimensions, lambda, model fingerprint and a payload checksum.
Both sides need the identical checkpoint. Coding runs on CPU. The `.b17` format
uses a reference adaptive binary arithmetic coder, **not original CABAC**.
Headers are larger than the paper's 49-bit header. Actual rates are measured
honestly and labeled `balle2017-reference-not-CABAC`. Model weights are shared
in advance and excluded from image bitrate, as in the paper.

## Evaluate

```powershell
python -m balle2017.evaluate --checkpoints runs/balle2017/gray-lambda512/best.pt --test-manifest data/kodak.txt --output runs/balle2017/kodak --jpeg-quality 10 20 40 60 80 --jpeg2000-rates 80 40 20 10
```

Outputs: `per_image.csv`, `summary.json`, `protocol.json`, reconstructed PNGs
and actual compressed streams. Each decoded image is checked against the
rounded network output. Reports include estimated bpp, payload bpp, whole-file
bpp, PSNR, luma PSNR, chroma PSNR (RGB), and luma MS-SSIM. Summaries average
per-image metrics at fixed lambda or classical codec setting.

JPEG uses 4:2:0 subsampling for RGB. JPEG2000 uses Pillow's OpenJPEG backend;
arguments are compression ratios, not target bitrates. Availability depends on
the Pillow build. Compare curves at matched rates, not quality-setting numbers.
PNG previews are clipped and rounded; metrics use raw float reconstructions.
`--skip-ms-ssim` is for quick development checks, not final reports.

For per-image classical files near the learned file size, add
`--matched-classical`. The search reports the remaining bitrate mismatch and
saves classical files/reconstructions. To generate plots, comparisons and a report:

```powershell
python -m balle2017.summarize_results --evaluation runs/balle2017/imagenette-gray-lambda512-20261102/kodak-best --training-run runs/balle2017/imagenette-gray-lambda512-20261102
```

The initial 24-image evaluation is complete; see [RESULTS.md](RESULTS.md).

## Authors' pretrained transforms

[Numeric MATLAB weights](https://www.cns.nyu.edu/~lcv/iclr2017/parameters/)
can be imported without executing downloaded pickle files:

```powershell
python -m balle2017.import_weights --directory balle2017/references/weights/gray/lambda-000512 --lambda 512 --output runs/imported-gray512.pt
```

The released transform files do not include learned spline density tables.
Imported checkpoints therefore have bitrate evaluation disabled until a density
is fitted on separate training/calibration data. Transform-only validation uses
`Balle2017.encode/decode` directly. Do not fit probabilities on the test set.

Independent numerical check against the released parameter arrays:

```powershell
python -m balle2017.verify_reference --directory balle2017/references/weights/gray/lambda-000512
```

This compares PyTorch with independently written NumPy analysis/synthesis;
it does not execute MATLAB. See [VALIDATION.md](VALIDATION.md) for measured results.

## Implementation map

| File | Responsibility |
| --- | --- |
| `layers.py` | DCT convolutions, projected GDN/IGDN |
| `model.py` | Three-stage transforms, released boundary convention, weight import |
| `density.py` | Ten-knot-per-unit linear spline marginals |
| `train.py` | Noise relaxation, separate optimizers, validation, checkpoints |
| `arithmetic.py` | Reference symbol decisions and adaptive arithmetic coding |
| `codec.py` | File container, fingerprint, checksum, decoding |
| `data.py` | Manifests, preprocessing and image I/O |
| `metrics.py`, `evaluate.py` | Metrics, classical baselines, per-image reports |

Next research milestones: validate imported transforms against the released
MATLAB program; recover/implement and validate original CABAC; lock the training
manifest and metric conventions; train multiple lambdas; compare original and
reproduced curves with every deviation recorded.
