import unittest
import numpy as np
import torch
from balle2017.arithmetic import encode_latents, decode_latents
from balle2017.codec import compress, decompress
from balle2017.layers import DCTConv, GDN, dct_basis
from balle2017.model import Balle2017, round_latents, symmetric_pad, boundary_padding
from balle2017.density import SplineDensity
from balle2017.train import objective


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_dct_and_filter_projection(self):
        basis = dct_basis(9)
        torch.testing.assert_close(basis@basis.T, torch.eye(9), atol=1e-6, rtol=1e-5)
        conv = DCTConv(3, 4, 5, 2)
        conv.coefficients.data.mul_(3)
        conv.project()
        w = conv.basis.T@conv.coefficients@conv.basis
        torch.testing.assert_close(w.square().sum((1, 2, 3)), torch.ones(4))

    def test_gdn_matches_numpy_and_constraints(self):
        norm = GDN(3)
        x = torch.randn(1, 3, 2, 2)
        beta = (norm.beta_prime.square()-norm.offset).detach().numpy()
        gamma = (norm.gamma_prime.square()-norm.offset).detach().numpy()
        a = x.numpy()
        expected = a/np.sqrt(beta[None, :, None, None]+np.einsum('ij,bjhw->bihw', gamma, a*a))
        np.testing.assert_allclose(norm(x).detach().numpy(), expected, rtol=1e-5)
        norm.gamma_prime.data[0, 1] = -1
        norm.project()
        torch.testing.assert_close(norm.gamma_prime, norm.gamma_prime.T)
        self.assertGreaterEqual(float(norm.gamma_prime.min().detach()), norm.lower)

    def test_boundary_matches_numpy_symmetric(self):
        x = torch.arange(5*7).reshape(1, 1, 5, 7).float()
        pads = boundary_padding(5, 7)
        expected = np.pad(x.numpy(), ((0, 0), (0, 0), pads[0], pads[1]), mode='symmetric')
        np.testing.assert_array_equal(symmetric_pad(x).numpy(), expected)

    def test_rounding_and_training_gradient(self):
        x = torch.tensor([-1.5, -.5, .5, 1.5], requires_grad=True)
        torch.testing.assert_close(round_latents(x), torch.tensor([-2., -1., 1., 2.]))
        round_latents(x).sum().backward()
        torch.testing.assert_close(x.grad, torch.zeros(4))
        model = Balle2017(channels=4, radius=100)
        source = torch.rand(1, 1, 33, 47)
        output = model(source)
        loss, _, _ = objective(output, source, 32)
        density_loss = -model.density(output['y_hat'].detach()).mean()
        loss.backward()
        self.assertGreater(float(model.analysis[0].coefficients.grad.abs().sum()), 0)
        self.assertIsNone(model.density.values.grad)
        density_loss.backward()
        self.assertGreater(float(model.density.values.grad.abs().sum()), 0)

    def test_spline_normalization_interpolation_and_support(self):
        d = SplineDensity(2, radius=3)
        area = torch.trapezoid(d.values, dx=.1, dim=1)
        torch.testing.assert_close(area, torch.ones(2))
        y = torch.tensor([[[[.05]], [[.15]]]], requires_grad=True)
        d(y).sum().backward()
        self.assertTrue(torch.isfinite(y.grad).all())
        with self.assertRaises(ValueError):
            d(torch.full((1, 2, 1, 1), 4.))

    def test_spline_expansion_preserves_density_and_checkpoint(self):
        d = SplineDensity(2, radius=3)
        y = torch.tensor([[[[.05]], [[.15]]]])
        original = d.values.detach().clone()
        self.assertTrue(d.expand_to(4, margin=2))
        self.assertEqual(d.radius, 6)
        expected = torch.nn.functional.pad(original, (30, 30), value=1e-9)
        expected /= torch.trapezoid(expected, dx=.1, dim=1)[:, None]
        torch.testing.assert_close(d.values, expected)
        torch.testing.assert_close(torch.trapezoid(d.values, dx=.1, dim=1), torch.ones(2))
        self.assertFalse(d.expand_to(1, margin=2))
        model = Balle2017(channels=4, radius=1)
        output = model(torch.rand(1, 1, 33, 47), adapt_support=True)
        self.assertTrue(output['support_expanded'])
        restored = Balle2017(**model.config)
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored.density.values, model.density.values)

    def test_arithmetic_modes_signs_escape_and_adaptation(self):
        rng = np.random.default_rng(17)
        integers = np.arange(-3, 4)
        for center in (0, 3, 6):
            probabilities = np.full((3, 7), .01)
            probabilities[:, center] = .94
            q = rng.integers(-100, 101, (3, 13, 17), dtype=np.int32)
            q[:, :3] = integers[center]
            data = encode_latents(q, integers, probabilities)
            decoded = decode_latents(data, q.shape, integers, probabilities)
            np.testing.assert_array_equal(q, decoded)

    def test_codec_odd_sizes_header_identity_and_corruption(self):
        model = Balle2017(channels=4).eval()
        for channels, h, w in [(1, 33, 47), (1, 1, 1), (3, 32, 48)]:
            if channels == 3:
                model = Balle2017(image_channels=3, channels=4).eval()
            x = torch.rand(1, channels, h, w)
            with torch.no_grad():
                expected = model(x)['x_hat']
                data = compress(model, x, 512)
                actual, metadata = decompress(model, data)
            self.assertEqual(actual.shape, x.shape)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            with self.assertRaises(ValueError):
                decompress(model, data[:-1])
            other = Balle2017(image_channels=channels, channels=4).eval()
            with self.assertRaises(ValueError):
                decompress(other, data)


if __name__ == '__main__':
    unittest.main()
