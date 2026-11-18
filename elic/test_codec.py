import unittest

import torch

from balle2017.arithmetic import Arithmetic
from .codec import code_values, compress, decompress
from .model import ELICSmall, GROUPS


class ELICCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(17)

    def test_tail_symbols_round_trip(self):
        mean = torch.tensor([0.25, -0.25, 0., 0., 0., 0.])
        scale = torch.tensor([0.11, 0.11, 1., 2., 4., 8.])
        values = mean+torch.tensor([-92., 71., -32., 32., -4., 5.])
        for logistic in (False, True):
            coder = Arithmetic()
            code_values(coder, mean, scale, values, logistic=logistic)
            decoded = code_values(Arithmetic(coder.finish()), mean, scale, logistic=logistic)
            torch.testing.assert_close(decoded, values, atol=0, rtol=0)

    @torch.no_grad()
    def test_nonanchors_cannot_influence_their_own_prediction(self):
        model = ELICSmall(width=8).eval()
        hyper = torch.randn(1, 2*sum(GROUPS), 4, 4)

        def trace(change_nonanchors):
            parameters = []

            def visit(index, mask, mean, scale):
                parameters.append((mean.clone(), scale.clone()))
                return mean + (1 if change_nonanchors and index == 0 and len(parameters) == 2 else 0)

            model.traverse(hyper, visit)
            return parameters

        original, changed = trace(False), trace(True)
        for i in range(2):
            for left, right in zip(original[i], changed[i]):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
        self.assertFalse(torch.equal(original[2][0], changed[2][0]))

    @torch.no_grad()
    def test_independent_file_decode_at_odd_size(self):
        encoder = ELICSmall(width=8).eval()
        decoder = ELICSmall(width=8).eval()
        decoder.load_state_dict(encoder.state_dict())
        image = torch.rand(1, 3, 33, 67)
        stream = compress(encoder, image)
        expected = encoder(image)
        actual = decompress(decoder, stream)
        for key in ('x_hat', 'y_hat'):
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        self.assertEqual(actual['x_hat'].shape, image.shape)
        self.assertEqual(sum(actual[key] for key in ('header_bytes', 'z_bytes', 'y_bytes')), len(stream))
        with self.assertRaises(ValueError):
            decompress(decoder, stream[:-1])
