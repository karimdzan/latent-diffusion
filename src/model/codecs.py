from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.blocks import AttnBlock, DownSample, ResnetBlock, UpSample


class Encoder(nn.Module):
    def __init__(
        self,
        channels: int,
        channel_multipliers: List[int],
        n_resnet_blocks: int,
        in_channels: int,
        z_channels: int,
        norm_groups: int = 32,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        channels_list = [m * channels for m in [1] + channel_multipliers]

        self.conv_in = nn.Conv2d(in_channels, channels, 3, stride=1, padding=1)
        self.down = nn.ModuleList(
            [
                self._build_down_block(
                    channels_list[i],
                    channels_list[i + 1],
                    n_resnet_blocks,
                    norm_groups,
                    norm_eps,
                    i != len(channel_multipliers) - 1,
                )
                for i in range(len(channel_multipliers))
            ]
        )

        self.mid = nn.Sequential(
            ResnetBlock(channels, channels, norm_groups=norm_groups, norm_eps=norm_eps),
            AttnBlock(channels, norm_groups, norm_eps),
            ResnetBlock(channels, channels, norm_groups=norm_groups, norm_eps=norm_eps),
        )

        self.norm_out = nn.GroupNorm(
            num_groups=norm_groups, num_channels=channels, eps=norm_eps
        )
        self.conv_out = nn.Conv2d(channels, 2 * z_channels, 3, stride=1, padding=1)

    def _build_down_block(
        self,
        current_channels,
        out_channels,
        n_resnet_blocks,
        norm_groups,
        norm_eps,
        use_downsample,
    ):
        resnet_blocks = nn.ModuleList(
            [
                ResnetBlock(
                    current_channels,
                    out_channels,
                    norm_groups=norm_groups,
                    norm_eps=norm_eps,
                )
                for _ in range(n_resnet_blocks)
            ]
        )

        down = nn.Module()
        down.block = resnet_blocks
        down.downsample = DownSample(out_channels) if use_downsample else nn.Identity()

        return down

    def forward(self, img: torch.Tensor):
        x = self.conv_in(img)

        for down in self.down:
            for block in down.block:
                x = block(x)
            x = down.downsample(x)

        x = self.mid(x)

        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        channels: int,
        channel_multipliers: List[int],
        n_resnet_blocks: int,
        out_channels: int,
        z_channels: int,
        norm_groups: int = 32,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        num_resolutions = len(channel_multipliers)

        channels_list = [m * channels for m in channel_multipliers]

        channels = channels_list[-1]

        self.conv_in = nn.Conv2d(z_channels, channels, 3, stride=1, padding=1)

        self.mid = nn.Sequential(
            ResnetBlock(channels, channels, norm_groups=norm_groups, norm_eps=norm_eps),
            AttnBlock(channels, norm_groups, norm_eps),
            ResnetBlock(channels, channels, norm_groups=norm_groups, norm_eps=norm_eps),
        )

        self.up = nn.ModuleList(
            [
                self._build_up_block(
                    channels_list[i], n_resnet_blocks, norm_groups, norm_eps, i != 0
                )
                for i in reversed(range(num_resolutions))
            ]
        )

        self.norm_out = nn.GroupNorm(
            num_groups=norm_groups, num_channels=channels, eps=norm_eps
        )
        self.conv_out = nn.Conv2d(channels, out_channels, 3, stride=1, padding=1)

    def _build_up_block(
        self, channels, n_resnet_blocks, norm_groups, norm_eps, use_upsample
    ):
        resnet_blocks = nn.ModuleList(
            [
                ResnetBlock(
                    channels, channels, norm_groups=norm_groups, norm_eps=norm_eps
                )
                for _ in range(n_resnet_blocks + 1)
            ]
        )

        up = nn.Module()
        up.block = resnet_blocks
        up.upsample = UpSample(channels) if use_upsample else nn.Identity()

        return up

    def forward(self, z: torch.Tensor):
        h = self.conv_in(z)

        h = self.mid(h)

        for up in reversed(self.up):
            for block in up.block:
                print(h.shape)
                h = block(h)
            h = up.upsample(h)

        h = self.norm_out(h)
        h = F.silu(h)
        img = self.conv_out(h)

        return img
