"""Generate static research plots, matched-rate comparisons and a readable report."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import load_image

LEARNED = 'balle2017-reference-not-CABAC'


def font(size):
    try:
        return ImageFont.truetype('C:/Windows/Fonts/arial.ttf', size)
    except OSError:
        return ImageFont.load_default()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation', required=True)
    p.add_argument('--training-run', required=True)
    args = p.parse_args()
    root, run = Path(args.evaluation), Path(args.training_run)
    with (root/'per_image.csv').open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    numeric = ['actual_bpp', 'estimated_bpp', 'payload_bpp', 'psnr', 'luma_ms_ssim', 'rate_mismatch_percent']
    for row in rows:
        for key in numeric:
            if row.get(key):
                row[key] = float(row[key])
    summary = json.loads((root/'summary.json').read_text())
    neural = [r for r in rows if r['method'] == LEARNED]
    if len(neural) != 24 or len({r['image'] for r in neural}) != 24:
        raise ValueError('Report expects a complete 24-image, one-checkpoint Kodak evaluation.')
    groups = {method: [r for r in rows if r['method'] == method]
              for method in (LEARNED, 'jpeg-matched', 'jpeg2000-matched')}
    lookup = {method: {r['image']:r for r in group} for method,group in groups.items()}
    for method in ('jpeg-matched', 'jpeg2000-matched'):
        if set(lookup[method]) != set(lookup[LEARNED]):
            raise ValueError('Incomplete matched-rate classical comparison.')
    averages = {method: {key:float(np.mean([r[key] for r in group]))
                        for key in ('actual_bpp', 'psnr', 'luma_ms_ssim')} for method,group in groups.items()}
    differences = {}
    for method in ('jpeg-matched', 'jpeg2000-matched'):
        gap = [r['psnr']-lookup[method][r['image']]['psnr'] for r in neural]
        differences[method] = {'mean_psnr_difference_learned_minus_classical':float(np.mean(gap)),
            'learned_psnr_wins':sum(g > 0 for g in gap),
            'learned_ms_ssim_wins':sum(r['luma_ms_ssim'] > lookup[method][r['image']]['luma_ms_ssim'] for r in neural),
            'mean_absolute_rate_mismatch_percent':float(np.mean([abs(r['rate_mismatch_percent']) for r in groups[method]])),
            'max_absolute_rate_mismatch_percent':float(max(abs(r['rate_mismatch_percent']) for r in groups[method]))}

    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    for ax, metric, label in zip(axes, ['psnr', 'luma_ms_ssim'], ['PSNR (dB)', 'Luma MS-SSIM']):
        for method, name, color in [('jpeg', 'JPEG', '#c47a16'), ('jpeg2000', 'JPEG2000', '#21806b')]:
            points = sorted([r for r in summary if r['method'] == method], key=lambda r:r['actual_bpp'])
            ax.plot([r['actual_bpp'] for r in points], [r[metric] for r in points], 'o-', color=color, label=name)
        r = averages[LEARNED]
        ax.scatter(r['actual_bpp'], r[metric], s=90, marker='s', color='#3c50bb', zorder=4, label='Our model: one operating point')
        ax.set(xlabel='Actual bits per pixel (whole file)', ylabel=label)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Kodak grayscale: mean of 24 images, eight-pixel border removed')
    fig.savefig(root/'rate_quality.png', dpi=170)
    plt.close(fig)

    metrics = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines() if line.strip()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout='constrained')
    steps = [r['step'] for r in metrics]
    axes[0].plot(steps, [r['validation_loss'] for r in metrics], color='#3c50bb')
    axes[0].set(xlabel='Training step', ylabel='Validation R + lambda D', title='Held-out validation loss')
    axes[1].plot(steps, [-10*np.log10(r['validation_squared_error_per_pixel']) for r in metrics], color='#21806b')
    axes[1].set(xlabel='Training step', ylabel='PSNR from pooled validation MSE (dB)', title='Validation reconstruction trend')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(root/'training_validation.png', dpi=170)
    plt.close(fig)

    # Deterministic worst / median / best by learned-minus-JPEG2000 PSNR.
    ordered = sorted(neural, key=lambda r:r['psnr']-lookup['jpeg2000-matched'][r['image']]['psnr'])
    chosen = [ordered[0], ordered[len(ordered)//2], ordered[-1]]
    model_root = root/'model-00-lambda-512'
    cell_w, cell_h, margin = 380, 325, 20
    sheet = Image.new('RGB', (margin*2+4*cell_w, 65+len(chosen)*cell_h), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 15), 'Matched-rate reconstructions | worst / median / best PSNR gap vs JPEG2000', fill='#20252b', font=font(22))
    details = Image.new('RGB', (margin*2+4*300, 65+len(chosen)*360), 'white')
    detail_draw = ImageDraw.Draw(details)
    detail_draw.text((margin, 15), 'Identical central 128 x 128 crops, enlarged 2x (nearest neighbor)', fill='#20252b', font=font(22))
    selections = []
    for row_index, row in enumerate(chosen):
        image_index = neural.index(row)
        stem = f'{image_index:03d}-{Path(row["image"]).stem}'
        original = (load_image(row['image'], 1, 8)[0].numpy()*255+.5).clip(0,255).astype(np.uint8)
        original = Image.fromarray(original)
        variants = [('Original', original, None),
                    ('Learned', Image.open(model_root/f'{stem}.png'), row),
                    ('JPEG', Image.open(model_root/'jpeg-matched'/f'{stem}.png'), lookup['jpeg-matched'][row['image']]),
                    ('JPEG2000', Image.open(model_root/'jpeg2000-matched'/f'{stem}.png'), lookup['jpeg2000-matched'][row['image']])]
        for column, (name, image, metric) in enumerate(variants):
            x, y = margin+column*cell_w, 65+row_index*cell_h
            preview = image.convert('RGB')
            preview.thumbnail((cell_w-15, 250), Image.Resampling.LANCZOS)
            sheet.paste(preview, (x, y+45))
            caption = f'{Path(row["image"]).stem} | {name}'
            value = f'{metric["actual_bpp"]:.3f} bpp | {metric["psnr"]:.2f} dB' if metric else 'Grayscale source'
            draw.text((x, y), caption, fill='#20252b', font=font(17))
            draw.text((x, y+22), value, fill='#4b5563', font=font(15))
            left, top = (image.width-128)//2, (image.height-128)//2
            crop = image.crop((left, top, left+128, top+128)).resize((256,256), Image.Resampling.NEAREST).convert('RGB')
            dx, dy = margin+column*300, 65+row_index*360
            details.paste(crop, (dx, dy+52))
            detail_draw.text((dx, dy), caption, fill='#20252b', font=font(17))
            detail_draw.text((dx, dy+23), value, fill='#4b5563', font=font(15))
        selections.append({'image': row['image'], 'learned_minus_jpeg2000_psnr':row['psnr']-lookup['jpeg2000-matched'][row['image']]['psnr']})
    sheet.save(root/'reconstructions.png')
    details.save(root/'detail_crops.png')
    analysis = {'averages':averages, 'paired_comparisons':differences, 'display_selection':selections,
                'mean_estimated_bpp':float(np.mean([r['estimated_bpp'] for r in neural])),
                'mean_payload_bpp':float(np.mean([r['payload_bpp'] for r in neural])),
                'mean_header_bpp':float(np.mean([r['actual_bpp']-r['payload_bpp'] for r in neural]))}
    (root/'analysis.json').write_text(json.dumps(analysis, indent=2), encoding='utf-8')
    names = {LEARNED:'Our learned model (reference coder)', 'jpeg-matched':'JPEG, matched per image', 'jpeg2000-matched':'JPEG2000, matched per image'}
    table = '\n'.join(f'| {names[m]} | {a["actual_bpp"]:.4f} | {a["psnr"]:.2f} | {a["luma_ms_ssim"]:.4f} |' for m,a in averages.items())
    comparison = '\n'.join(f'- Against {names[m]}: mean PSNR difference **{v["mean_psnr_difference_learned_minus_classical"]:+.2f} dB**; learned model wins PSNR on {v["learned_psnr_wins"]}/24 and MS-SSIM on {v["learned_ms_ssim_wins"]}/24 images. Mean absolute rate mismatch {v["mean_absolute_rate_mismatch_percent"]:.2f}%, maximum {v["max_absolute_rate_mismatch_percent"]:.2f}%.' for m,v in differences.items())
    report = f'''# First Kodak evaluation — 2026-11-02

All 24 images evaluated; all 24 files decoded exactly to the rounded neural
forward-pass reconstruction. Best checkpoint: step 9,750, selected using held-out
Imagenette validation; training ran 10,000 steps. Grayscale, 128 channels, lambda 512.

## Average results

| Method | Actual bpp | Mean PSNR (dB) | Mean luma MS-SSIM |
| --- | ---: | ---: | ---: |
{table}

{comparison}

The mean estimated neural rate is {analysis['mean_estimated_bpp']:.4f} bpp;
actual payload is {analysis['mean_payload_bpp']:.4f} bpp and container overhead
is {analysis['mean_header_bpp']:.4f} bpp. Whole-file rate includes the custom header.

## Figures

![Classical rate-quality curves and our single point](rate_quality.png)
![Validation trend](training_validation.png)
![Reconstruction comparisons](reconstructions.png)
![Enlarged center crops](detail_crops.png)

The displayed images are selected by the worst, median and best learned-minus-
JPEG2000 PSNR gap; they are not handpicked successes. Exact full-resolution
reconstructions, matched classical files and learned bitstreams are in
`model-00-lambda-512/`.

## Interpretation limits

- This is one operating point, so no learned rate-distortion curve or BD-rate
  claim is possible. The classical curves average fixed quality/ratio settings.
- Matched comparisons use real encoded files at the closest searched size;
  rates are approximate, with their mismatches reported above.
- Training used 4,726 filtered, full-resolution Imagenette images (ten classes),
  not the paper's ImageNet subset. The model is only an initial 10,000-step run.
- The arithmetic coder is our adaptive reference implementation, not original
  CABAC. This is not a historical-performance reproduction claim.
- Eight pixels are removed from each Kodak edge. All methods use the same
  grayscale crop. Classical codecs consume rounded 8-bit grayscale; errors are
  measured against original floating-point luma. Neural metrics use raw floating
  outputs, while PNG previews are clipped and rounded.
- MS-SSIM uses pytorch-msssim; parity with the legacy implementation is unverified.
- Table PSNR is the mean of per-image PSNRs. Validation trend uses PSNR derived
  from mean validation MSE, which is a different aggregation.

Machine-readable results: `per_image.csv`, `summary.json`, `analysis.json`,
`protocol.json`. The training run, manifests and dataset hashes remain alongside
the documented experiment. Test images did not select the checkpoint.
'''
    (root/'REPORT.md').write_text(report, encoding='utf-8')
    print(json.dumps(analysis, indent=2))


if __name__ == '__main__':
    main()
