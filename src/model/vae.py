from typing import List

import torch
import torch.nn as nn

from src.model.codecs import Decoder, Encoder


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def mode(self):
        return self.mean


"""_____________ VAE _____________"""


class VAE(nn.Module):
    def __init__(
        self,
        img_ch: int,
        enc_channels: int,
        ch_mult: List[int],
        resnet_blocks: int,
        latent_dim: int,
        norm_groups: int,
        eps: float,
        scaling_factor: float = 0.18215,
    ):
        super(VAE, self).__init__()

        self.encoder = Encoder(
            channels=enc_channels,
            channel_multipliers=ch_mult,
            n_resnet_blocks=resnet_blocks,
            in_channels=img_ch,
            z_channels=latent_dim,
            norm_groups=norm_groups,
            norm_eps=eps,
        )

        self.decoder = Decoder(
            channels=enc_channels,
            channel_multipliers=ch_mult,
            n_resnet_blocks=resnet_blocks,
            out_channels=img_ch,
            z_channels=latent_dim,
            norm_groups=norm_groups,
            norm_eps=eps,
        )

        self.scaling_factor = scaling_factor
        self.quant_conv = nn.Conv2d(2 * latent_dim, 2 * latent_dim, 1)
        self.post_quantizer = nn.Conv2d(latent_dim, latent_dim, 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(h)
        return posterior

    def decode(self, z):
        z = self.post_quantizer(z / self.scaling_factor)
        dec = self.decoder(z)
        return dec

    def forward(self, x):
        posterior = self.encode(x)
        z = posterior.sample() * self.scaling_factor
        dec = self.decode(z)
        return {
            "recon_x": dec,
            "x": z,
            "logvar": posterior.logvar,
            "mean": posterior.mean,
        }
