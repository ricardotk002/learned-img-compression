"""One lambda per run; Adam transforms and independent projected spline SGD."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import TrainingImages, read_manifest, load_image
from .model import Balle2017
from .codec import load_checkpoint


def objective(output, x, lmbda):
    pixels = x.shape[0]*x.shape[-2]*x.shape[-1]
    rate = -output['likelihoods'].log2().sum()/pixels
    # Equivalent per-pixel scaling of Eq. 7: sum squared channel errors per pixel.
    distortion = (output['x_hat']-x).square().sum()/pixels
    return rate + lmbda*distortion, rate, distortion


@torch.no_grad()
def validate(model, images, channels, device, lmbda, crop=256):
    model.eval()
    values = []
    for path in images:
        x = load_image(path, channels)[None].to(device)
        if crop:
            if min(x.shape[-2:]) < crop:
                raise ValueError('Validation image is smaller than requested crop.')
            top, left = [(size-crop)//2 for size in x.shape[-2:]]
            x = x[..., top:top+crop, left:left+crop]
        loss, rate, distortion = objective(model(x), x, lmbda)
        values.append([float(loss), float(rate), float(distortion)])
    model.train()
    return np.mean(values, axis=0).tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-manifest', required=True)
    p.add_argument('--val-manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--lambda', dest='lmbda', type=int, default=512)
    p.add_argument('--mode', choices=['gray', 'rgb'], default='gray')
    p.add_argument('--channels', type=int, help='Default: 128 gray / 192 RGB; 256 for high-rate experiments.')
    p.add_argument('--radius', type=int, default=30)
    p.add_argument('--steps', type=int, default=1000000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--density-lr', type=float, default=0.01)
    p.add_argument('--saturation-fraction', type=float, default=0.01)
    p.add_argument('--validate-every', type=int, default=1000)
    p.add_argument('--plateau-validations', type=int, default=5)
    p.add_argument('--seed', type=int, default=2017)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--resume')
    p.add_argument('--dataset-description', default='Unspecified image manifest')
    p.add_argument('--max-training-images', type=int)
    p.add_argument('--validation-crop', type=int, default=256,
                   help='Deterministic center crop for held-out validation; 0 uses full images.')
    args = p.parse_args()
    if not 1 <= args.lmbda <= 65535 or min(args.steps, args.validate_every, args.batch_size) < 1:
        p.error('Positive steps/batch/validation interval and 16-bit lambda required.')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    image_channels = 1 if args.mode == 'gray' else 3
    dataset = TrainingImages(args.train_manifest, image_channels, saturation_fraction=args.saturation_fraction)
    if args.max_training_images:
        dataset.accepted = dataset.accepted[:args.max_training_images]
    validation = read_manifest(args.val_manifest)
    if set(dataset.images) & set(validation):
        raise ValueError('Training and validation manifests overlap.')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    model = Balle2017(image_channels, args.channels or (128 if image_channels == 1 else 192), args.radius)
    saved = None
    if args.resume:
        model, saved = load_checkpoint(args.resume)
        if saved['lambda'] != args.lmbda or model.config['image_channels'] != image_channels:
            raise ValueError('Resume lambda/mode differs from run.')
    model.to(args.device).train()
    transforms = [v for k, v in model.named_parameters() if not k.startswith('density.')]
    optimizer = torch.optim.Adam(transforms, lr=args.lr)
    density_optimizer = torch.optim.SGD(model.density.parameters(), lr=args.density_lr)
    start = 0
    best_rate = best_distortion = best_loss = float('inf')
    stale_rate = stale_distortion = 0
    if saved and 'optimizer' in saved:
        optimizer.load_state_dict(saved['optimizer'])
        density_optimizer.load_state_dict(saved['density_optimizer'])
        start = saved['step']
        torch.set_rng_state(saved['torch_rng'])
        if torch.cuda.is_available() and saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        if 'best_loss' in saved:
            best_loss, best_rate, best_distortion = saved['best_loss'], saved['best_rate'], saved['best_distortion']
            stale_rate, stale_distortion = saved['stale_rate'], saved['stale_distortion']
    if not torch.cuda.is_available() and args.device.startswith('cuda'):
        raise RuntimeError('CUDA requested but unavailable.')
    manifests = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in
                 [('train', args.train_manifest), ('validation', args.val_manifest)]}
    run_info = {'arguments': vars(args), 'config': model.config, 'torch': str(torch.__version__),
                'manifest_sha256': manifests, 'preprocessing': dataset.report(),
                'reproduction_status': 'Baseline reconstruction; original implementation parity unverified'}
    (root/'run.json').write_text(json.dumps(run_info, indent=2), encoding='utf-8')
    iterator = iter(loader)
    started = time.monotonic()
    observed_maximum = 0
    (root/'status.json').write_text(json.dumps({'status': 'running', 'step': start, 'target_steps': args.steps}), encoding='utf-8')
    for step in range(start+1, args.steps+1):
        try:
            x = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x = next(iterator)
        x = x.to(args.device)
        optimizer.zero_grad(set_to_none=True)
        density_optimizer.zero_grad(set_to_none=True)
        output = model(x, adapt_support=True)
        observed_maximum = max(observed_maximum, float(output['y_hat'].detach().abs().max()))
        if output['support_expanded']:
            density_optimizer = torch.optim.SGD(model.density.parameters(), lr=args.density_lr)
            print(json.dumps({'step': step, 'spline_radius_expanded': model.density.radius}), flush=True)
        loss, rate, distortion = objective(output, x, args.lmbda)
        density_loss = -model.density(output['y_hat'].detach()).mean((0, 2, 3)).sum()
        if not torch.isfinite(loss + density_loss):
            raise FloatingPointError('Non-finite training objective.')
        loss.backward()
        density_loss.backward()
        optimizer.step()
        model.project()
        density_optimizer.step()
        model.density.project()
        if step % 1000000 == 0:
            # Periodic check plus early overflow protection; unspecified original
            # heuristic reconstructed conservatively, without shrinking support.
            if model.density.expand_to(observed_maximum):
                model.config['radius'] = model.density.radius
                density_optimizer = torch.optim.SGD(model.density.parameters(), lr=args.density_lr)
            observed_maximum = 0
        if step == 1 or step % 100 == 0:
            print(json.dumps({'step': step, 'loss': float(loss.detach()), 'bpp': float(rate.detach()),
                              'squared_error_per_pixel': float(distortion.detach()),
                              'steps_per_second': (step-start)/(time.monotonic()-started)}), flush=True)
            (root/'status.json').write_text(json.dumps({'status': 'running', 'step': step,
                'target_steps': args.steps, 'loss': float(loss.detach()), 'bpp': float(rate.detach()),
                'elapsed_seconds': time.monotonic()-started}), encoding='utf-8')
        if step % args.validate_every == 0 or step == args.steps:
            vloss, vrate, vdist = validate(model, validation, image_channels, args.device, args.lmbda, args.validation_crop)
            stale_rate = 0 if vrate < best_rate else stale_rate+1
            stale_distortion = 0 if vdist < best_distortion else stale_distortion+1
            best_rate, best_distortion = min(best_rate, vrate), min(best_distortion, vdist)
            improved = vloss < best_loss
            best_loss = min(best_loss, vloss)
            if min(stale_rate, stale_distortion) >= args.plateau_validations:
                for group in optimizer.param_groups:
                    group['lr'] = max(1e-7, group['lr']/10)
                stale_rate = stale_distortion = 0
            report = {'step': step, 'validation_loss': vloss, 'validation_bpp': vrate,
                      'validation_squared_error_per_pixel': vdist, 'lr': optimizer.param_groups[0]['lr']}
            print(json.dumps(report), flush=True)
            with (root/'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(report)+'\n')
            state = {'model': model.state_dict(), 'config': model.config, 'lambda': args.lmbda,
                     'step': step, 'optimizer': optimizer.state_dict(),
                     'density_optimizer': density_optimizer.state_dict(), 'torch_rng': torch.get_rng_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                     'best_loss': best_loss, 'best_rate': best_rate, 'best_distortion': best_distortion,
                     'stale_rate': stale_rate, 'stale_distortion': stale_distortion,
                     'density_fitted': True, 'provenance': run_info}
            torch.save(state, root/'last.pt.tmp')
            (root/'last.pt.tmp').replace(root/'last.pt')
            if improved:
                torch.save(state, root/'best.pt.tmp')
                (root/'best.pt.tmp').replace(root/'best.pt')
            (root/'run.json').write_text(json.dumps(run_info, indent=2), encoding='utf-8')
    (root/'status.json').write_text(json.dumps({'status': 'completed', 'step': args.steps,
        'target_steps': args.steps, 'elapsed_seconds': time.monotonic()-started}), encoding='utf-8')


if __name__ == '__main__':
    main()
