"""Import authors' numeric MATLAB transform files. No trained density is bundled."""
import argparse
import torch
from .model import Balle2017, import_matlab


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--lambda', dest='lmbda', type=int, required=True)
    p.add_argument('--mode', choices=['gray', 'rgb'], default='gray')
    p.add_argument('--channels', type=int)
    args = p.parse_args()
    model = Balle2017(1 if args.mode == 'gray' else 3,
                      args.channels or (128 if args.mode == 'gray' else 192))
    import_matlab(model, args.directory)
    torch.save({'model': model.state_dict(), 'config': model.config, 'lambda': args.lmbda,
                'density_fitted': False, 'source': 'authors MATLAB transforms; spline density missing'}, args.output)
    print('Imported transforms. Entropy density is NOT fitted; rate evaluation remains disabled.')


if __name__ == '__main__':
    main()
