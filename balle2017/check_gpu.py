"""Verify CUDA with a synthetic 256x256 baseline training step, not a research run."""
import argparse
import json
from pathlib import Path
import torch
from .model import Balle2017
from .train import objective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/gpu-check.json')
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Install CUDA-enabled PyTorch in the system-wide Python environment.')
    if args.batch_size < 1:
        parser.error('Batch size must be positive.')
    torch.set_num_threads(2)
    torch.manual_seed(2017)
    torch.cuda.reset_peak_memory_stats()
    model = Balle2017().to('cuda').train()
    transforms = [p for name, p in model.named_parameters() if not name.startswith('density.')]
    optimizer = torch.optim.Adam(transforms, lr=1e-4)
    density_optimizer = torch.optim.SGD(model.density.parameters(), lr=0.01)
    source = torch.rand(args.batch_size, 1, 256, 256, device='cuda')
    before = model.analysis[0].coefficients.detach().clone()
    output = model(source)
    loss, rate, distortion = objective(output, source, 512)
    loss.backward()
    density_loss = -model.density(output['y_hat'].detach()).mean()
    density_loss.backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
        raise AssertionError('Missing or non-finite gradients.')
    optimizer.step()
    model.project()
    density_optimizer.step()
    model.density.project()
    if torch.equal(before, model.analysis[0].coefficients):
        raise AssertionError('Training step did not update encoder filters.')
    model.eval()
    with torch.no_grad():
        reconstructed = model(source)['x_hat']
    if reconstructed.shape != source.shape or not torch.isfinite(reconstructed).all():
        raise AssertionError('GPU reconstruction invalid.')
    torch.cuda.synchronize()
    report = {'torch': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(0), 'device': str(reconstructed.device),
              'batch_size': args.batch_size, 'input_shape': list(source.shape),
              'channels': 128, 'training_loss': float(loss.detach()),
              'rate_bpp': float(rate.detach()), 'squared_error_per_pixel': float(distortion.detach()),
              'peak_allocated_MiB': torch.cuda.max_memory_allocated()/2**20,
              'status': 'forward, backward, both optimizers, projection and rounded evaluation passed',
              'scope': 'Synthetic correctness check; not a trained compression baseline.'}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
