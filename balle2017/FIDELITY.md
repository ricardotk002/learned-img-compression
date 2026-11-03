# Historical fidelity audit

Target: close reproduction of the 2017 paper, **not** the 2018 factorized model.
The current implementation is an initial reconstruction. Passing software tests
does not demonstrate reproduction of the published compression performance.

Sources:
- [Paper, equations 1-17 and appendices 6.1-6.3](https://arxiv.org/pdf/1611.01704)
- [Authors' project page](https://www.cns.nyu.edu/~lcv/iclr2017/)
- [Released encoder](https://www.cns.nyu.edu/~lcv/iclr2017/compress.m)
- [Released decoder](https://www.cns.nyu.edu/~lcv/iclr2017/uncompress.m)

| Component | Current implementation | Remaining qualification |
| --- | --- | --- |
| Analysis | 9/5/5 kernels, strides 4/2/2, GDN after ALL three | Follows paper; numerical reference comparison pending |
| Synthesis | IGDN BEFORE each upsample/convolution, kernels 5/5/9 | Independent weights, no output sigmoid |
| Centering | Additional latent biases | Matches released MATLAB stages 03/00 |
| Widths | Gray 128; RGB 192; configurable 256 | Exact high-rate RGB configurations need release inspection |
| GDN | beta/gamma = prime squared minus 2^-10; project prime >= 2^-5; symmetric gamma | Follows appendix; initialization is a choice |
| Filters | Orthonormal DCT coefficients; unit norm after optimizer step | Follows appendix; initialization is a choice |
| Quantization | Unit-width uniform noise in training; rounding at evaluation | MATLAB ties away from zero |
| Probability | Shared spatially per channel; linear spline with ten knots/unit | No Gaussian/2018 entropy bottleneck replacement |
| Density training | Separate projected SGD on negative density (Eq. 17), sum over channels and average over batch/space | SGD learning rate not specified; chosen 0.01 |
| Spline range | Grow-only expansion with two-unit margin, positive tail padding, normalization and SGD rebinding; million-step check | Implemented heuristic differs from unspecified original; no shrinkage |
| Transform optimizer | Adam 1e-4, reductions to 1e-7 when both rate/distortion stagnate | Validation cadence/patience and batch size are choices |
| Input range/loss | [0,1]; bpp + lambda * summed channel error/pixel | Confirm original RGB lambda normalization before numeric comparison |
| Training set | First run uses 4,726 filtered full-resolution Imagenette training images; 256 held-out validation images | User-approved ten-class ImageNet subset; not original 6,507-image training set |
| Preprocessing | Saturation filter, pixel noise, scale < .75, 256 crop | Saturation threshold, scale distribution, resampling filter and grayscale conversion order not specified; choices documented |
| Image boundaries | Edge-inclusive symmetric padding, valid convolutions, center crop | Matches released MATLAB procedure; authors say original Python boundaries differed |
| Binarization | Mode, sign, candidate decisions, Exp-Golomb escape; channel/raster scan | Reference reconstruction of Fig. 9 |
| Entropy engine | Adaptive integer binary arithmetic with per-channel contexts | **NOT H.264 CABAC**; context initialization, adaptation and engine differ |
| Header | Version, dimensions, lambda, model hash, CRC | Custom container, **not original 49-bit header**; report payload and full rates |
| Test set | All 24 Kodak PNGs downloaded, default eight-pixel border removal at final evaluation | Test split never used for checkpoint selection |
| Color metrics | JPEG full-range YCbCr, luma/chroma errors, luma MS-SSIM | Exact original conversion precision needs validation |
| MS-SSIM | pytorch-msssim library | Parity with authors' 2003 implementation remains unverified |
| Classical codecs | Pillow JPEG (4:2:0 RGB), OpenJPEG JPEG2000 ratios | Library versions and original encoder flags differ; logged protocol |
| Published results | No claims yet | Need trained checkpoints, reference transforms/coder validation and R-D curves |

## Required before claiming close historical reproduction

1. Validate imported weights against numeric outputs from the authors' MATLAB
   implementation, then quantify boundary differences from original Python.
2. Implement and verify the original CABAC engine/context behavior and header,
   or explicitly classify the result as a method reproduction with a coder deviation.
3. Freeze training/validation/test manifests and record image hashes; acknowledge
   that an identical 6,507-image subset is not currently recoverable.
4. Validate reconstructed spline support adaptation and verify RGB lambda normalization.
5. Validate MS-SSIM/color conventions and reproduce multi-lambda R-D curves,
   including actual coded rates and matched-rate visual comparisons.

Original unspecified details cannot be honestly described as exact matches.
Published performance is an experimental target, not a consequence of matching
the layer sizes.
