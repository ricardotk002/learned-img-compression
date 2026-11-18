# Learned Image Compression

PyTorch implementations of Ballé 2017, the Ballé 2018 scale hyperprior, conditional variable-rate compression, ELIC-sm, and full ELIC, plus an MLIC++ reproduction notebook.

## Setup

```text
python -m pip install -r requirements.txt
```

Training and evaluation commands are available for [Ballé 2017](balle2017/README.md), [Ballé 2018](balle2018/README.md), [variable rate](variable_rate/README.md), and [ELIC](elic/README.md). The new hyperprior methods use Python 3.12 with PyTorch 2.6, torchvision 0.21, and CompressAI 1.2.8; exact pilot package versions and optional test/notebook dependencies are documented in their method directories. All generated data, checkpoints, logs, and results for these methods stay in their respective method-local, Git-ignored directories.

## Kodak results

Mean scores on all 24 Kodak images after an eight-pixel crop on each edge. Bitrates include file headers. JPEG and JPEG2000 settings were selected per image to approximate the corresponding learned file size, using the same color mode.

| Run | Codec | Actual bpp ↓ | Luma PSNR (dB) ↑ | Luma MS-SSIM ↑ |
| --- | --- | ---: | ---: | ---: |
| Grayscale | **Ballé 2017** | 0.5543 | 29.60 | 0.9464 |
| | JPEG | 0.5543 | 31.55 | 0.9784 |
| | JPEG2000 | 0.5528 | 34.04 | 0.9783 |
| RGB, small | **ELIC-sm** | 0.5136 | 26.33 | 0.9194 |
| | JPEG | 0.5127 | 30.61 | 0.9687 |
| | JPEG2000 | 0.5133 | 29.41 | 0.9390 |
| RGB, full | **ELIC** | 0.5444 | 26.96 | 0.9319 |
| | JPEG | 0.5437 | 30.92 | 0.9726 |
| | JPEG2000 | 0.5442 | 29.61 | 0.9425 |

All three checkpoints trailed their matched-size JPEG and JPEG2000 baselines on both mean luma metrics. Full ELIC gained 0.62 dB over ELIC-sm at 0.031 additional bpp. The Ballé run encodes grayscale, while both ELIC runs encode RGB, so the learned bitrates cover different color information.

## MLIC++ results

[`mlic++/mlic_reproduction.ipynb`](mlic++/mlic_reproduction.ipynb) evaluates the official [MLIC++](https://github.com/JiangWeibeta/MLIC) checkpoints and a short DIV2K training run on Kodak. Scores are RGB means over all 24 images, without the edge crop used above.

| Run | λ | bpp ↓ | PSNR (dB) ↑ | MS-SSIM ↑ |
| --- | ---: | ---: | ---: | ---: |
| **MLIC++**, official weights | 0.0018 | 0.1075 | 29.16 | 0.9278 |
| | 0.0035 | 0.1747 | 30.72 | 0.9511 |
| | 0.0067 | 0.2721 | 32.33 | 0.9673 |
| | 0.0130 | 0.4129 | 34.17 | 0.9788 |
| | 0.0250 | 0.5923 | 35.82 | 0.9857 |
| | 0.0483 | 0.8026 | 37.32 | 0.9901 |
| **MLIC++**, 40-epoch run | 0.0483 | 0.9234 | 27.21 | 0.9470 |

The official weights reproduce the published PSNR BD-rate of −15.14% versus VTM. The 40-epoch run trailed the official λ = 0.0483 checkpoint by 10.1 dB at a higher rate.

## Scale-hyperprior and variable-rate pilots

These experiments use **full-image RGB** PSNR and MS-SSIM on all 24 Kodak images, without the edge crop used in the Ballé 2017/ELIC table. Bitrates include file headers and both entropy streams. Both models trained from scratch on the same 760/40 DIV2K split for approximately 90 minutes on an Apple M4 GPU.

| Model / control | Actual bpp ↓ | RGB PSNR (dB) ↑ | RGB MS-SSIM ↑ |
| --- | ---: | ---: | ---: |
| **Ballé 2018 scale hyperprior**, λ=0.0067 | 0.41388 | 28.874 | 0.95314 |
| **Variable rate**, quality 0, Δ=1 | 0.23584 | 26.739 | 0.92576 |
| Variable rate, quality 1, Δ=1 | 0.30250 | 27.572 | 0.93958 |
| Variable rate, quality 2, Δ=1 | 0.35861 | 27.933 | 0.94711 |
| Variable rate, quality 3, Δ=1 | 0.39694 | 28.172 | 0.95074 |

One variable-rate checkpoint supports 20 tested quality/quantization settings spanning 0.11852–0.66696 bpp. All 480 saved files independently matched deterministic inference, with no observed rate reversals across the tested controls. The fixed-rate pilot remains more PSNR-efficient near 0.4 bpp. Equal wall-clock time shared across four objectives is not equal training per operating point, and these short pilots are not converged paper reproductions. Only one fixed-rate hyperprior point exists, so a baseline-family BD-rate is not available.

The [baseline notebook](balle2018/notebooks/baseline_analysis.ipynb) and [variable-rate notebook](variable_rate/notebooks/variable_rate_analysis.ipynb) read local artifacts. Trained weights, datasets, generated images, and full evaluation outputs are not stored in Git.

## Published comparisons

| Model | Published Kodak result | Reproduction notes |
| --- | --- | --- |
| Ballé 2017 | Generally better rate–distortion performance than JPEG and JPEG2000, with a clearer MS-SSIM advantage. [§4 and Appendix §6.3](https://arxiv.org/pdf/1611.01704) | The single measured point trailed both codecs. Training used Imagenette for 10,000 steps and a reference arithmetic coder in place of the paper's CABAC-based coder. |
| ELIC-sm | −1.07% PSNR BD-rate and −5.59% MS-SSIM BD-rate versus VVC. [Table 2 and supplementary Table 3](https://openaccess.thecvf.com/content/CVPR2022/html/He_ELIC_Efficient_Learned_Image_Compression_With_Unevenly_Grouped_Space-Channel_Contextual_CVPR_2022_paper.html) | The checkpoint trailed both matched classical codecs. Training used Imagenette for 2,000 steps, with a simplified hyperlatent prior and reference arithmetic coder. |
| Full ELIC | −7.88% PSNR BD-rate and −12.73% MS-SSIM BD-rate versus VVC. [Table 2 and supplementary Table 3](https://openaccess.thecvf.com/content/CVPR2022/html/He_ELIC_Efficient_Learned_Image_Compression_With_Unevenly_Grouped_Space-Channel_Contextual_CVPR_2022_paper.html) | The larger transform improved luma quality at a higher rate after the same 2,000-step training budget and still trailed both matched classical codecs. |
| MLIC++ | −15.14% PSNR BD-rate versus VTM, computed from the [published Kodak curve](https://github.com/JiangWeibeta/MLIC/tree/4aa9d2a/results/kodak). | The official checkpoints matched the published curve. A 40-epoch DIV2K run at λ = 0.0483 reached 27.21 dB at 0.9234 bpp, 10.1 dB below the official checkpoint. |

Published ELIC BD-rates summarize curves relative to VVC. Each local Ballé and ELIC model provides one measured operating point, which is insufficient for BD-rate. The papers trained on selected ImageNet subsets with different training schedules.
