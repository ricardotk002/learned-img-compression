"""Compare imported transforms to an independent NumPy reading of released equations.

This is not a substitute for running the authors' MATLAB executable or verifying
the missing original Python/CABAC implementation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import torch
from .model import Balle2017, import_matlab, boundary_padding


def numpy_analysis(x, directory):
    pads = boundary_padding(*x.shape[-2:])
    y = np.pad(x, ((0, 0), (0, 0), pads[0], pads[1]), mode='symmetric')
    for stage in range(3):
        p = loadmat(directory/f'analysis-{stage:02d}.mat')
        h = p['h'].astype(np.float64)
        stride = int(p['strides'].reshape(-1)[0])
        windows = np.lib.stride_tricks.sliding_window_view(y, h.shape[-2:], axis=(-2, -1))
        windows = windows[:, :, ::stride, ::stride]
        y = np.einsum('bcxyij,ocij->boxy', windows, h, optimize=True)
        y += p['c'].reshape(1, -1, 1, 1)
        norm = np.einsum('bchw,co->bohw', y*y, p['gamma'], optimize=True)
        y /= np.sqrt(norm+p['beta'].reshape(1, -1, 1, 1))
    return y+loadmat(directory/'analysis-03.mat')['c'].reshape(1, -1, 1, 1)


def numpy_synthesis(q, directory, size):
    y = q+loadmat(directory/'synthesis-00.mat')['c'].reshape(1, -1, 1, 1)
    for stage in range(1, 4):
        p = loadmat(directory/f'synthesis-{stage:02d}.mat')
        norm = np.einsum('bchw,co->bohw', y*y, p['gamma'], optimize=True)
        y = y*np.sqrt(norm+p['beta'].reshape(1, -1, 1, 1))
        h = p['h'].astype(np.float64)
        stride = int(p['strides'].reshape(-1)[0])
        height = (y.shape[-2]-1)*stride+h.shape[-2]
        width = (y.shape[-1]-1)*stride+h.shape[-1]
        out = np.zeros((1, h.shape[1], height, width), dtype=np.float64)
        for i in range(h.shape[-2]):
            for j in range(h.shape[-1]):
                out[:, :, i:i+stride*y.shape[-2]:stride, j:j+stride*y.shape[-1]:stride] += np.einsum(
                    'bchw,co->bohw', y, h[:, :, i, j], optimize=True)
        y = out+p['c'].reshape(1, -1, 1, 1)
    top, left = [(a-b)//2 for a, b in zip(y.shape[-2:], size)]
    return y[:, :, top:top+size[0], left:left+size[1]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', required=True)
    p.add_argument('--output', default='runs/reference_verification.json')
    args = p.parse_args()
    directory = Path(args.directory)
    torch.set_num_threads(2)
    # Double precision minimizes errors introduced by DCT conversion.
    model = Balle2017().double().eval()
    for layer in list(model.analysis)+list(model.synthesis):
        if hasattr(layer, 'basis'):
            n = layer.basis.shape[0]
            k = torch.arange(n, dtype=torch.float64)[:, None]
            basis = torch.cos(torch.pi/n*(torch.arange(n, dtype=torch.float64)+.5)*k)*(2/n)**.5
            basis[0] /= 2**.5
            layer.basis.copy_(basis)
    import_matlab(model, directory)
    x = np.random.default_rng(2017).uniform(0, 1, (1, 1, 33, 47))
    expected_y = numpy_analysis(x, directory)
    with torch.no_grad():
        actual_y = model.encode(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(actual_y, expected_y, rtol=2e-5, atol=2e-5)
    q = np.sign(expected_y)*np.floor(np.abs(expected_y)+.5)
    expected_x = numpy_synthesis(q, directory, x.shape[-2:])
    with torch.no_grad():
        actual_x = model.decode(torch.from_numpy(q), x.shape[-2:]).numpy()
    np.testing.assert_allclose(actual_x, expected_x, rtol=2e-5, atol=2e-5)
    report = {'reference': str(directory), 'analysis_max_abs_error': float(np.abs(actual_y-expected_y).max()),
              'synthesis_max_abs_error': float(np.abs(actual_x-expected_x).max()),
              'rounded_latents_equal': bool(np.array_equal(np.sign(actual_y)*np.floor(np.abs(actual_y)+.5), q)),
              'scope': 'Independent NumPy equations using real released weights; NOT executed MATLAB or original Python parity.'}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
