# Implementation checks (2026-10-29)

These checks establish implementation behavior, not reproduction of the paper's
rate-distortion results. All training checks used synthetic noise images; there
is no trained research baseline yet.

## Completed

- Seven unittest cases pass: DCT orthogonality/filter normalization, GDN formula
  and projection, edge-inclusive symmetric padding, rounding versus noise
  gradients and optimizer separation, spline normalization/support checks,
  arithmetic coding with both signs/modes/tail escapes, and file round trips
  including odd/tiny/RGB dimensions, wrong-model detection and corruption.
- Full 128-channel grayscale architecture: one optimizer step at 256x256 and
  a rounded validation pass completed with finite objectives.
- Small 8-channel synthetic integration run: training, checkpoint save/load,
  resume with optimizer state, image compression, independent file decompression,
  CSV/JSON reports, PSNR, luma MS-SSIM, JPEG and OpenJPEG JPEG2000 completed.
  Decoded floating reconstruction exactly matched rounded model inference.
- Authors' grayscale lambda=512 `.mat` parameters imported successfully.
- Independent NumPy analysis/synthesis using those real weights on a seeded
  33x47 synthetic input agreed with the double-precision PyTorch transforms:
  analysis maximum absolute error 8.21e-8; synthesis 1.34e-8; rounded latents
  identical. This validates our reading of the released parameter arrays;
  it is **not** a comparison with an executed MATLAB/Python reference program.

Local scratch results are ignored under `runs/smoke/` and
`runs/reference_verification.json`. Local downloaded sources/weights are ignored
under `balle2017/references/`.

## Environment

| Dependency | Verified version |
| --- | --- |
| Python | 3.12.10 |
| PyTorch | 2.14.1+cpu |
| NumPy | 2.4.6 |
| SciPy | 1.18.1 |
| Pillow | 13.0.0 |
| pytorch-msssim | 1.0.0 |

These versions describe the original 2026-10-29 CPU checks. The former
project-local `.venv` was removed on 2026-11-01 at the project owner's request.

## System-wide GPU setup (2026-11-01)

The system-wide Python 3.12 installation now uses PyTorch **2.11.0+cu128**
(CUDA runtime 12.8). `torch.cuda.is_available()` returns `True` and detects
the NVIDIA GeForce RTX 3070 Ti Laptop GPU. pytorch-msssim 1.0.0 is installed
system-wide as well. setuptools 81.0.0 satisfies this PyTorch build's dependency.
Use `python` directly; do not create or activate a project virtual environment.

`python -m balle2017.check_gpu` passed on CUDA with the full 128-channel grayscale
model and a 256x256 synthetic image (batch size 1). Forward pass, backward pass,
both optimizers, parameter projections, and rounded evaluation succeeded with
finite gradients/reconstructions and updated encoder weights. The measured
peak allocated tensor memory was 77.01 MiB (not total driver/workspace memory).
The report is stored locally at `runs/gpu-check.json`.

This is a correctness check; no long training run or natural-image experiment
was started by the setup operation.

## Not validated

Original H.264 CABAC behavior, original Python boundary behavior, original
training-set membership, million-step spline range adaptation, original metric
code parity, convergence on natural images, long GPU training runs, and published R-D
curves. These remain explicit gates in [FIDELITY.md](FIDELITY.md).

## Natural-image launch (2026-11-02)

All eight current unit tests pass, including the new spline expansion,
normalization and checkpoint-shape checks. A 20-step natural-image GPU pilot
completed; held-out validation and checkpointing passed.

Downloaded full-resolution Imagenette and 24 Kodak PNGs. Prepared content-disjoint
splits: 4,726 training / 256 validation / 24 test images with source/image hashes.
The main 10,000-step grayscale/128-channel/lambda512/batch8 run was launched in
the background using system-wide CUDA PyTorch. It passed its first 250-step
validation and saved `last.pt` and `best.pt`. At that checkpoint, estimated
validation bpp was 0.6222 and validation loss 10.7712 on held-out 256 center crops.
These are preliminary validation values, not Kodak results or actual coded rates.
See [FIRST_RUN.md](FIRST_RUN.md). Check the live logs for current status;
this document does not imply that the process is still running at a later date.

## Completed training and Kodak evaluation (2026-11-02)

The initial run completed 10,000 steps in 1,402.34 seconds. The best validation
checkpoint is step 9,750. All 24 Kodak images were encoded, decoded and checked
against rounded inference with exact floating-output equality. Per-image JPEG
and JPEG2000 matching used actual compressed bytes; maximum rate mismatch was
below 1.5%. Mean learned results: 0.55435 actual bpp, 29.60 dB PSNR, 0.94645 luma
MS-SSIM. This first model trails both classical baselines. See [RESULTS.md](RESULTS.md)
for the full comparison and limitations. Plots and sample crops were visually
inspected. No additional training was launched as part of this evaluation.
