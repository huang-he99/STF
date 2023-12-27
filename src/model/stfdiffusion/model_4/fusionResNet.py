import math
from functools import partial


import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased=False))
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


# sinusoidal positional embeds


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LearnedSinusoidalPosEmb(nn.Module):
    """following @crowsonkb 's lead with learned sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.ReLU(), nn.Linear(time_emb_dim, dim_out * 2))

        self.proj1 = nn.Conv2d(dim, dim_out, 3, padding=1)
        # self.norm1 = nn.GroupNorm(groups, dim_out)
        self.act1 = nn.ReLU()

        self.proj2 = nn.Conv2d(dim_out, dim_out, 3, padding=1)
        # self.norm2 = nn.GroupNorm(groups, dim_out)
        # self.act2 = nn.ReLU()

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        time_emb = self.mlp(time_emb)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        scale_shift = time_emb.chunk(2, dim=1)

        h = self.proj1(x)
        # h = self.norm1(h)
        scale, shift = scale_shift
        h = h * (scale + 1) + shift
        h = self.act1(h)

        h = self.proj2(h)
        # h = self.norm2(h)
        # h = self.act2(h)

        return h + self.res_conv(x)


class FusionResnet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        channels=24,
        depth=4,
        learned_sinusoidal_cond=False,
        learned_sinusoidal_dim=16,
        self_condition=False,
    ):
        super().__init__()
        self.self_condition = self_condition
        # time embeddings
        time_dim = dim * 4
        self.channels = channels
        self.learned_sinusoidal_cond = learned_sinusoidal_cond

        if learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers

        init_dim = init_dim if init_dim is not None else dim
        self.init_conv = nn.Conv2d(channels, init_dim, 3, padding=1)

        dims = [init_dim, *(dim,) * depth]

        self.resnet = nn.ModuleList()

        for i in range(depth):
            self.resnet.append(ResnetBlock(dims[i], dims[i + 1], time_emb_dim=time_dim))

        self.final_res_block = ResnetBlock(dim, dim, time_emb_dim=time_dim)

        self.tail_conv = nn.Conv2d(dim, out_dim, 3, 1, 1)

    def forward(
        self,
        coarse_img_01,
        coarse_img_02,
        fine_img_01,
        noisy_fine_img_02,
        time,
        x_self_cond=None,
    ):
        x = torch.cat(
            (coarse_img_01, coarse_img_02, fine_img_01, noisy_fine_img_02), dim=1
        )

        x = self.init_conv(x)

        r = x.clone()
        t = self.time_mlp(time)
        for resnet in self.resnet:
            x = resnet(x, time_emb=t)
        x = x + r
        x = self.final_res_block(x, time_emb=t)
        x = self.tail_conv(x)
        return x
