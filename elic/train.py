"""Train ELIC on an RGB patch manifest and save the latest checkpoint."""
import argparse
import itertools
import json
import math
from time import perf_counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from balle2017.data import TrainingImages, load_image, read_manifest
from .model import ELIC


@torch.no_grad()
def validate(model, paths, device, lmbda):
    model.eval()
    totals = dict(loss=0., psnr=0., y_bpp=0., z_bpp=0.)
    for path in paths:
        image = load_image(path, 3)
        h, w = image.shape[-2:]
        top, left = max(0, (h-256)//2), max(0, (w-256)//2)
        x = image[None, :, top:top+256, left:left+256].to(device)
        result = model(x)
        mse = (result['x_hat']-x).square().mean().item()
        y_bpp, z_bpp = (result[key].item() for key in ('y_bpp', 'z_bpp'))
        totals['loss'] += y_bpp+z_bpp+lmbda*255**2*mse
        totals['psnr'] += -10*math.log10(max(mse, 1e-12))
        totals['y_bpp'] += y_bpp
        totals['z_bpp'] += z_bpp
    model.train()
    return {key: value/len(paths) for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-manifest', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--variant', choices=('small', 'full'), default='small')
    parser.add_argument('--mixed-steps', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lambda', dest='lmbda', type=float, default=.015)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=2017)
    parser.add_argument('--val-manifest')
    parser.add_argument('--val-limit', type=int, default=32)
    parser.add_argument('--validate-every', type=int, default=250)
    args = parser.parse_args()
    if (min(args.steps, args.batch_size, args.val_limit, args.validate_every, args.lmbda) <= 0
            or not math.isfinite(args.lmbda) or not 0 <= args.mixed_steps <= args.steps):
        parser.error('Require positive finite run settings, with mixed steps within the run.')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    dataset = TrainingImages(args.train_manifest, channels=3, saturation_fraction=.05)
    validation = read_manifest(args.val_manifest) if args.val_manifest else []
    if set(validation) & set(dataset.images):
        parser.error('Training and validation manifests overlap.')
    validation = validation[:args.val_limit]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    batches = itertools.chain.from_iterable(iter(loader) for _ in itertools.count())
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    model = ELIC(variant=args.variant).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')
    metrics_path = args.output.with_suffix('.validation.jsonl')
    best_path = args.output.with_name(args.output.stem+'.best'+args.output.suffix)
    if any(path.exists() for path in (args.output, best_path, metrics_path)):
        parser.error('Choose a new output path for this run.')
    print(json.dumps({'train_images': len(dataset), 'validation_images': len(validation),
                      'target_steps': args.steps, 'device': args.device, 'variant': model.variant,
                      'parameters': sum(p.numel() for p in model.parameters())}), flush=True)
    started = perf_counter()
    for step in range(1, args.steps+1):
        x = next(batches).to(args.device)
        stage = 'mixed' if step <= args.mixed_steps else 'centered'
        result = model(x, stage)
        mse = (result['x_hat']-x).square().mean()
        loss = result['y_bpp']+result['z_bpp']+args.lmbda*255**2*mse
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite ELIC loss.')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            progress = {'step': step, 'stage': stage, 'loss': loss.item(), 'mse': mse.item(),
                        'y_bpp': result['y_bpp'].item(), 'z_bpp': result['z_bpp'].item(),
                        'elapsed_seconds': perf_counter()-started}
            if device.type == 'cuda':
                progress['peak_allocated_mib'] = torch.cuda.max_memory_allocated(device)/2**20
                progress['peak_reserved_mib'] = torch.cuda.max_memory_reserved(device)/2**20
            print(json.dumps(progress), flush=True)
        if step == args.steps or step % args.validate_every == 0:
            checkpoint = {'width': model.width, 'variant': model.variant, 'model': model.state_dict(), 'step': step,
                          'optimizer': optimizer.state_dict(), 'lambda': args.lmbda}
            if validation:
                metrics = validate(model, validation, args.device, args.lmbda)
                report = json.dumps({'step': step, 'validation': metrics})
                print(report, flush=True)
                with metrics_path.open('a', encoding='utf-8') as stream:
                    stream.write(report+'\n')
                if metrics['loss'] < best_loss:
                    best_loss = metrics['loss']
                    save_checkpoint(checkpoint, best_path)
            save_checkpoint(checkpoint, args.output)
    print(json.dumps({'status': 'completed', 'step': args.steps}), flush=True)


def save_checkpoint(checkpoint, path):
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


if __name__ == '__main__':
    main()
