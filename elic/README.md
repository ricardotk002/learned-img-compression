# ELIC baseline

Reproduction of [Efficient Learned Image Compression with Unevenly Grouped Space-Channel Contextual Adaptive Coding](https://openaccess.thecvf.com/content/CVPR2022/html/He_ELIC_Efficient_Learned_Image_Compression_With_Unevenly_Grouped_Space-Channel_Contextual_CVPR_2022_paper.html).

Install the repository dependencies and run commands from the repository root.

```text
python -m pip install -r requirements.txt
python -m unittest elic.test_codec -v
```

Prepare the shared Imagenette and Kodak manifests if needed:

```text
python -m balle2017.prepare_data --download --saturation-fraction 0.05
```

Train an RGB ELIC-sm checkpoint. `--mixed-steps` controls when training switches
from noisy context and straight-through rounding to mean-centered reconstruction.
These step counts are for a short pilot. Use `--device cpu` to train on a CPU.

```text
python -m elic.train --train-manifest data/train.txt --output runs/elic/pilot.pt --steps 20 --mixed-steps 10 --device cuda
```

For a longer run with held-out validation:

```text
python -m elic.train --train-manifest data/train.txt --val-manifest data/val.txt --output runs/elic/longer/last.pt --steps 2000 --mixed-steps 1000 --batch-size 2 --device cuda
```

By default, validation uses centered crops of the first 32 validation images
every 250 steps. It logs RGB PSNR and estimated rates for both streams in
`last.validation.jsonl`. `last.best.pt` is selected by validation loss.
Use a new output path for each run.

To train the full ELIC transform with three residual bottlenecks per stage and
attention, select `--variant full`. Start with batch size 2 for a GPU pilot:

```text
python -m elic.train --variant full --train-manifest data/train.txt --val-manifest data/val.txt --output runs/elic/full/last.pt --steps 2000 --mixed-steps 1000 --batch-size 2 --device cuda
```

Both variants use the same reference entropy model. Compression and evaluation
select the variant from the checkpoint. Existing ELIC-sm checkpoints are supported.

Compress and independently decompress an image with the same checkpoint:

```text
python -m elic compress --checkpoint runs/elic/pilot.pt --input data/kodak/kodim01.png --output runs/elic/sample.elic
python -m elic decompress --checkpoint runs/elic/pilot.pt --input runs/elic/sample.elic --output runs/elic/sample.png
```

File coding runs on the CPU. The encoder reports whole-file bits per pixel and
separate main stream, hyperlatent stream, and header sizes. This implementation
uses a reference arithmetic coder with quantized scale tables and a learned
factorized logistic prior for hyperlatents. Coding speed and rate differ from
the paper's implementation.

## Evaluate on Kodak

```text
python -m elic.evaluate --checkpoint runs/elic/longer/last.best.pt --test-manifest data/kodak.txt --output runs/elic/longer/kodak
```

The evaluator removes eight pixels from each border, checks the decoded latents
and reconstruction against rounded inference, and reports whole-file bpp,
estimated bpp, RGB and luma PSNR, and luma MS-SSIM. It measures JPEG and JPEG2000
at several settings and searches for a file size close to ELIC's for each image.
Per-image metrics, method averages, reconstructions, and compressed files are
written to the output directory. The existing grayscale Ballé 2017 results use
the same luma metrics and crop, while ELIC and its classical baselines encode RGB.
