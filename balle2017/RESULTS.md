# Initial Kodak results (2026-11-02)

[Full report and comparison figures](results/imagenette-gray-lambda512-20261102/REPORT.md)
are included in the repository, together with machine-readable evaluation tables.

The first trained model works as a file codec, but currently underperforms JPEG
and JPEG2000 at approximately matched rates. This experiment does not reproduce
the paper's performance.

All 24 Kodak images were evaluated in grayscale after removing eight pixels
from each edge. All compressed files decoded exactly to the rounded model's
floating reconstruction. Test images were not used to choose the checkpoint.

| Method | Mean actual bpp | Mean PSNR | Mean luma MS-SSIM |
| --- | ---: | ---: | ---: |
| Learned model, reference coder | 0.55435 | 29.60 dB | 0.94645 |
| JPEG, matched per image | 0.55429 | 31.55 dB | 0.97839 |
| JPEG2000, matched per image | 0.55279 | 34.04 dB | 0.97834 |

The learned model is behind JPEG by 1.95 dB on average and JPEG2000 by 4.45 dB.
It wins PSNR on 1/24 images against JPEG and 0/24 against JPEG2000. Mean absolute
per-image bitrate mismatches are 0.70% and 0.41%, respectively, with maximums
below 1.5%. These are real encoded files, not interpolated quality measurements.

Estimated learned rate is 0.56996 bpp; actual arithmetic payload is 0.54839 bpp;
custom header overhead is 0.00596 bpp. Actual whole-file sizes are reported above.

## Visual inspection

Inspected full-image comparisons and enlarged identical center crops selected by
the worst, median and best PSNR gap against JPEG2000 (kodim23, kodim02, kodim01).
The learned model introduces visible granular/high-frequency artifacts in smooth
regions and softens some fine edges/textures. Classical codec artifacts differ,
but their reconstructions are substantially cleaner in smooth regions in this
comparison. These observations do not by themselves identify the training or
implementation cause; they warrant investigation before claiming convergence.

## Scope

- Training: 4,726 filtered Imagenette images, 256 validation images, 10,000 steps.
- Evaluated checkpoint: step 9,750, best by held-out validation loss.
- Grayscale, 128 channels, lambda 512; one operating point only.
- Reference adaptive arithmetic coder, not original CABAC.
- Raw floating neural reconstructions are used for metrics; PNGs are clipped
  and rounded. Classical codecs consume 8-bit grayscale. Metric implementation
  and original training-set parity remain unverified.
- A single learned point cannot establish a rate-distortion curve or BD-rate.

Full outputs are under ignored local directory:
`runs/balle2017/imagenette-gray-lambda512-20261102/kodak-best/`.
Open `REPORT.md`, `rate_quality.png`, `training_validation.png`,
`reconstructions.png`, and `detail_crops.png`. Machine-readable results are
`per_image.csv`, `summary.json`, `analysis.json`, and `protocol.json`.
The curated repository snapshot excludes weights, datasets and per-image
bitstreams; those remain in the local run folder.

The next research step is to investigate the reconstruction artifacts and
training behavior, then use validation data to choose a longer controlled run.
Avoid using Kodak results to select among training checkpoints or tune the model.
