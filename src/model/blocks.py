import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum


class LinearAttnBlock(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        """Adopted from: https://huggingface.co/blog/annotated-diffusion"""
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class AttnBlock(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        """
        dim: int -- input size
        heads: int -- number of heads
        dim_head: int -- head dimensionality

        Adopted from: https://huggingface.co/blog/annotated-diffusion
        """
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class UpSample(nn.Module):
    def __init__(self, dim: int, dim_out: int, scale: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode="nearest"),
            nn.Conv2d(in_channels=dim, out_channels=dim_out, kernel_size=3, padding=1),
            nn.GroupNorm(max(1, dim_out // 4), dim_out),
        )

    def forward(self, x: torch.Tensor):
        return self.up(x)


class DownSample(nn.Module):
    def __init__(self, dim: int, dim_out: int, ks: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=ks),
            nn.Conv2d(in_channels=dim, out_channels=dim_out, kernel_size=3, padding=1),
            nn.GroupNorm(max(1, dim_out // 4), dim_out),
        )

    def forward(self, x: torch.Tensor):
        return self.down(x)


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization

    SF: do not use einops
    Adopted from https://huggingface.co/blog/annotated-diffusion
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-4

        weight = self.weight
        mean = torch.mean(weight, dim=[1, 2, 3], keepdim=True)
        var = torch.var(weight, unbiased=False, dim=[1, 2, 3], keepdim=True)
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class conv_block(nn.Module):
    """Adopted from: https://huggingface.co/blog/annotated-diffusion"""

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """
    https://arxiv.org/abs/1512.03385
    Adopted from: https://huggingface.co/blog/annotated-diffusion
    """

    def __init__(
        self, in_channels, out_channels, res_hidden=None, time_emb_dim=None, groups=8
    ):
        super().__init__()

        self.mlp = None
        if time_emb_dim is not None:
            self.mlp = nn.Sequential(
                nn.SiLU(), nn.Linear(time_emb_dim, out_channels * 2)
            )

        if not res_hidden:
            res_hidden = out_channels

        self.block1 = conv_block(in_channels, res_hidden, groups=groups)
        self.block2 = conv_block(res_hidden, out_channels, groups=groups)

        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            # time_emb = rearrange(time_emb, "b c -> b c 1 1")
            time_emb = time_emb.view(time_emb.shape[0], time_emb.shape[1], 1, 1)
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim, embed_param=10000):
        """
        embed_param - a magical parameter that everyone uses as 10'000
        """
        super().__init__()
        self.dim = dim
        self.T = embed_param

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(self.T) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
