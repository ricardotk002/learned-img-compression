# Learned Image Compression

PyTorch implementations of Ballé 2017, ELIC-sm, and full ELIC for training, file compression, and evaluation.

## Setup

```text
python -m pip install -r requirements.txt
```

Training and evaluation commands are available for [Ballé 2017](balle2017/README.md) and [ELIC](elic/README.md).

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

## Published comparisons

| Model | Published Kodak result | Reproduction notes |
| --- | --- | --- |
| Ballé 2017 | Generally better rate–distortion performance than JPEG and JPEG2000, with a clearer MS-SSIM advantage. [§4 and Appendix §6.3](https://arxiv.org/pdf/1611.01704) | The single measured point trailed both codecs. Training used Imagenette for 10,000 steps and a reference arithmetic coder in place of the paper's CABAC-based coder. |
| ELIC-sm | −1.07% PSNR BD-rate and −5.59% MS-SSIM BD-rate versus VVC. [Table 2 and supplementary Table 3](https://openaccess.thecvf.com/content/CVPR2022/html/He_ELIC_Efficient_Learned_Image_Compression_With_Unevenly_Grouped_Space-Channel_Contextual_CVPR_2022_paper.html) | The checkpoint trailed both matched classical codecs. Training used Imagenette for 2,000 steps, with a simplified hyperlatent prior and reference arithmetic coder. |
| Full ELIC | −7.88% PSNR BD-rate and −12.73% MS-SSIM BD-rate versus VVC. [Table 2 and supplementary Table 3](https://openaccess.thecvf.com/content/CVPR2022/html/He_ELIC_Efficient_Learned_Image_Compression_With_Unevenly_Grouped_Space-Channel_Contextual_CVPR_2022_paper.html) | The larger transform improved luma quality at a higher rate after the same 2,000-step training budget and still trailed both matched classical codecs. |

Published ELIC BD-rates summarize curves relative to VVC. Each local model provides one measured operating point, which is insufficient for BD-rate. The papers trained on selected ImageNet subsets with different training schedules.
