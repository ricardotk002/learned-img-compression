# First natural-image training experiment (2026-11-02)

This is a preliminary method-reproduction experiment, not an exact historical
reproduction. The owner approved Imagenette for this first run because official
ImageNet access requires an approved login and the paper's subset is unspecified.

## Data

- Full-resolution Imagenette v2 from the [maintainer's download](https://github.com/fastai/imagenette).
  The archive contains 9,469 training and 3,925 validation images from ten ImageNet
  classes. Never use the 320px/160px versions: our required downsampling would
  leave them too small for 256x256 crops.
- After size filtering and rejecting images with more than 5% of channel values
  at 0 or 255: **4,726 training images**. Use all qualifying supplied training
  images; do not add images from the supplied validation split merely to match
  the original paper's training count.
- **256 held-out validation images**, deterministic selection with seed 2017
  from Imagenette's supplied validation split, same eligibility filters.
- All **24 original-sized RGB Kodak PNGs** from
  [the lossless Kodak suite](https://r0k.us/graphics/kodak/), reserved for testing.
- Image byte hashes are checked across the selected splits. Full source,
  acceptance/rejection and SHA256 provenance lives in `data/dataset_provenance.json`.
- Class labels are ignored. Ten-class bias and changed sample count must be
  disclosed in the class report; these results do not establish ImageNet parity.

Manifest SHA256:

| Manifest | SHA256 |
| --- | --- |
| `data/train.txt` | `a28c0eee9cc290bbf98101c4a56ffe842f4481dadc7ba1cdac8e98d151b1b788` |
| `data/val.txt` | `15f31b7ba285171219228ba46f59991aaa02a5b2412554a747a9b4a01b2487bd` |
| `data/kodak.txt` | `2c8a0b9ed098181f93788c449e9af5c10e2ba0040bcca87039a99f84ca27a349` |

Archive SHA256: `6cbfac238434d89fe99e651496f0812ebc7a10fa62bd42d6874042bf01de4efd`.

## Choices for unspecified or unfinished details

1. Preprocessing: dequantize with U(-0.5/255,0.5/255) in RGB, sample a uniform
   scale from 256/min(H,W) to .75 (upper bound excluded), antialiased bicubic
   resize, random 256 crop, JPEG-weighted grayscale conversion. No flips or
   normalization to ImageNet classification means/stds.
2. Probability model: ten spline knots/unit; projected SGD with channel-summed
   negative likelihood (not negative log likelihood). Positive normalized knots.
3. Support adaptation: retain existing knots, pad positive tails and renormalize;
   grow at least 25% with a two-unit margin, including early overflow protection.
   Never shrink. Rebind the plain SGD optimizer after replacing its Parameter.
   This is an explicit reconstruction of an unspecified heuristic.
4. CABAC: retain the existing reference coder for eventual evaluation, label
   actual rates as non-CABAC, and report headers separately. The coder is not
   needed for the differentiable training objective and does not block training.
5. Validation: rounded latents on fixed 256 center crops, no pixel/latent noise,
   average per-image loss; choose best checkpoint by rate-distortion objective.
6. Evaluation: keep the documented raw float metrics, JPEG luma conversion,
   luma MS-SSIM and eight-pixel Kodak border removal. No claim of exact legacy
   metric implementation parity. Do not evaluate Kodak during training.

## Run configuration

From random initialization; grayscale; 128 channels; lambda 512; batch 8;
10,000 steps; Adam 1e-4 for transforms; SGD .01 for splines; seed 2017;
validation/checkpoints every 250 steps; CUDA in system-wide Python.

The full run starts fresh; a separate 20-step/64-image pilot was used only to
check training and checkpoint behavior. Pilot validation loss decreased from
60.21 at step 10 to 37.19 at step 20 on 16 validation images. These values are
software-readiness checks, not evidence of converged compression performance.

```powershell
.\balle2017\start_training.ps1
Get-Content runs/balle2017/imagenette-gray-lambda512-20261102/stdout.log -Tail 10
Get-Content runs/balle2017/imagenette-gray-lambda512-20261102/status.json
# Resume after interruption, using the latest complete checkpoint:
.\balle2017\start_training.ps1 -Resume
```

Keep the computer awake and powered for training to continue. The launcher
records its PID in `process.json`; a PID/status file alone does not prove the
process is still running. The latest checkpoint is atomically replaced after
validation. No automatic monitoring or scheduled follow-up is configured.
