"""ELIC transforms and a shared causal traversal for inference and coding."""
import torch
from torch import nn
from torch.nn import functional as F


GROUPS = (16, 16, 32, 64, 192)


def down(source, target, kernel=5):
    return nn.Conv2d(source, target, kernel, stride=2, padding=kernel//2)


def up(source, target, kernel=5):
    return nn.ConvTranspose2d(source, target, kernel, stride=2,
                              padding=kernel//2, output_padding=1)


class ResidualBottleneck(nn.Module):
    def __init__(self, width):
        super().__init__()
        hidden = width//2
        self.layers = nn.Sequential(nn.Conv2d(width, hidden, 1), nn.ReLU(),
                                    nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(hidden, width, 1))

    def forward(self, x):
        return x + self.layers(x)


class Attention(nn.Module):
    """Cheng-style gated residual attention used in full ELIC."""

    def __init__(self, channels):
        super().__init__()
        self.trunk = nn.Sequential(*(ResidualBottleneck(channels) for _ in range(3)))
        self.gate = nn.Sequential(*(ResidualBottleneck(channels) for _ in range(3)),
                                  nn.Conv2d(channels, channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x + self.trunk(x)*self.gate(x)


def ste_round(x):
    return x + (x.round() - x).detach()


def bin_mass(residual, scale, logistic=False):
    # Use symmetry to evaluate the lower tail and avoid subtracting two values near 1.
    distance = residual.abs()
    def cdf(value):
        return torch.sigmoid(value) if logistic else 0.5*torch.erfc(-value/2**0.5)
    upper = cdf((0.5-distance)/scale)
    lower = cdf((-0.5-distance)/scale)
    return (upper-lower).clamp_min(1e-9)


class SliceContext(nn.Module):
    def __init__(self, previous, width, latent_channels):
        super().__init__()
        self.channel = nn.Sequential(nn.Conv2d(previous, 2*width, 5, padding=2), nn.ReLU(),
                                     nn.Conv2d(2*width, 2*width, 5, padding=2)) if previous else None
        self.spatial = nn.Conv2d(width, 2*width, 5, padding=2)
        coordinates = torch.arange(5)
        self.register_buffer('mask', ((coordinates[:, None]+coordinates) % 2)[None, None])
        inputs = 2*latent_channels + (4 if previous else 2)*width
        self.aggregate = nn.Sequential(nn.Conv2d(inputs, 3*width, 1), nn.ReLU(),
                                       nn.Conv2d(3*width, 2*width, 1))

    def parameters_for(self, shared, anchors):
        if anchors is None:
            batch, _, h, w = shared[0].shape
            spatial = shared[0].new_zeros(batch, self.spatial.out_channels, h, w)
        else:
            spatial = F.conv2d(anchors, self.spatial.weight*self.mask,
                               self.spatial.bias, padding=2)
        mean, log_scale = self.aggregate(torch.cat([*shared, spatial], dim=1)).chunk(2, 1)
        return mean, F.softplus(log_scale).clamp_min(0.11)


class ELIC(nn.Module):
    def __init__(self, width=192, variant='small'):
        super().__init__()
        if width < 2:
            raise ValueError('Transform width must be at least two.')
        self.width = width
        if variant not in ('small', 'full'):
            raise ValueError(f'Unknown ELIC variant: {variant}')
        self.variant = variant
        latent = sum(GROUPS)
        full = variant == 'full'

        def residuals():
            return [ResidualBottleneck(width) for _ in range(3 if full else 1)]

        def attention(channels):
            return [Attention(channels)] if full else []

        self.analysis = nn.Sequential(down(3, width), *residuals(),
                                      down(width, width), *residuals(), *attention(width),
                                      down(width, width), *residuals(), down(width, latent),
                                      *attention(latent))
        self.synthesis = nn.Sequential(*attention(latent), up(latent, width), *residuals(),
                                       up(width, width), *attention(width), *residuals(),
                                       up(width, width), *residuals(), up(width, 3))
        self.hyper_analysis = nn.Sequential(nn.Conv2d(latent, width, 3, padding=1), nn.ReLU(),
                                            down(width, width), nn.ReLU(), down(width, width))
        self.hyper_synthesis = nn.Sequential(up(width, width), nn.ReLU(), up(width, width), nn.ReLU(),
                                             nn.Conv2d(width, 2*latent, 3, padding=1))
        self.z_mean = nn.Parameter(torch.zeros(1, width, 1, 1))
        self.z_log_scale = nn.Parameter(torch.zeros(1, width, 1, 1))
        self.contexts = nn.ModuleList(SliceContext(sum(GROUPS[:i]), size, latent)
                                      for i, size in enumerate(GROUPS))

    @property
    def z_scale(self):
        return F.softplus(self.z_log_scale).clamp_min(0.11)

    def analyze(self, x):
        h, w = x.shape[-2:]
        return self.analysis(F.pad(x, (0, -w % 64, 0, -h % 64)))

    def traverse(self, hyper, visit):
        """Visit each group twice, passing only decoder-available tensors."""
        h, w = hyper.shape[-2:]
        row = torch.arange(h, device=hyper.device)[:, None]
        column = torch.arange(w, device=hyper.device)
        anchor_mask = (row+column) % 2 == 0
        decoded = []
        for index, (context, width) in enumerate(zip(self.contexts, GROUPS)):
            shared = [hyper]
            if context.channel is not None:
                shared.append(context.channel(torch.cat(decoded, dim=1)))
            zeros = hyper.new_zeros(hyper.shape[0], width, h, w)
            mean, scale = context.parameters_for(shared, None)
            anchors = zeros.clone()
            anchors[..., anchor_mask] = visit(index, anchor_mask, mean[..., anchor_mask], scale[..., anchor_mask])
            mean, scale = context.parameters_for(shared, anchors)
            group = anchors.clone()
            group[..., ~anchor_mask] = visit(index, ~anchor_mask, mean[..., ~anchor_mask], scale[..., ~anchor_mask])
            decoded.append(group)
        return torch.cat(decoded, dim=1)

    def forward(self, x, stage='centered'):
        if stage not in ('mixed', 'centered'):
            raise ValueError('Unknown quantization stage.')
        y = self.analyze(x)
        z = self.hyper_analysis(y)
        centered_z = z-self.z_mean
        quantize = ste_round if self.training else torch.round
        z_hat = quantize(centered_z)+self.z_mean
        z_sample = centered_z + torch.rand_like(z)-0.5 if self.training else centered_z.round()
        parts = y.split(GROUPS, dim=1)
        rates = []

        def visit(index, mask, mean, scale):
            values = parts[index][..., mask]
            residual = values-mean
            sample = residual+torch.rand_like(residual)-0.5 if self.training else residual.round()
            rates.append(-bin_mass(sample, scale).log2().sum())
            if self.training and stage == 'mixed':
                return sample+mean
            return quantize(residual)+mean

        reconstructed = self.traverse(self.hyper_synthesis(z_hat), visit)
        synthesis_input = ste_round(y) if self.training and stage == 'mixed' else reconstructed
        h, w = x.shape[-2:]
        pixels = x.shape[0]*h*w
        return {'x_hat': self.synthesis(synthesis_input)[..., :h, :w], 'y_hat': reconstructed,
                'y_bpp': sum(rates)/pixels,
                'z_bpp': -bin_mass(z_sample, self.z_scale, logistic=True).log2().sum()/pixels}


class ELICSmall(ELIC):
    def __init__(self, width=192):
        super().__init__(width=width, variant='small')


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = ELIC(width=checkpoint['width'], variant=checkpoint.get('variant', 'small'))
    model.load_state_dict(checkpoint['model'])
    return model.eval(), checkpoint
