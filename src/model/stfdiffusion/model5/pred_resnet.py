import math

from functools import partial
from collections import namedtuple


import torch
from torch import nn, einsum
import torch.nn.functional as F


from einops import rearrange, reduce
from einops.layers.torch import Rearrange


# helpers functions

# small helper modules


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


# building block modules


# class Block(nn.Module):
#     def __init__(self, dim, dim_out, groups=8):
#         super().__init__()
#         self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
#         self.norm = nn.GroupNorm(groups, dim_out)
#         self.act = nn.SiLU()

#     def forward(self, x, scale_shift=None):
#         x = self.proj(x)
#         x = self.norm(x)

#         if exists(scale_shift):
#             scale, shift = scale_shift
#             x = x * (scale + 1) + shift

#         x = self.act(x)
#         return x


class NoisePredResBlock(nn.Module):
    def __init__(self, dim, time_emb_dim=None):
        super().__init__()
        self.mlp = nn.Sequential(nn.ReLU(), nn.Linear(time_emb_dim, dim * 2))

        self.conv_1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.norm_1 = nn.BatchNorm2d(dim)
        self.act_1 = nn.ReLU()

        self.conv_2 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.norm_2 = nn.BatchNorm2d(dim)

    def forward(self, x, time_emb=None):
        time_emb = self.mlp(time_emb)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        scale_shift = time_emb.chunk(2, dim=1)

        h = self.conv_1(x)
        h = self.norm_1(h)
        scale, shift = scale_shift
        h = h * (scale + 1) + shift
        h = self.act_1(h)

        h = self.conv_2(h)
        h = self.norm_2(h)

        return h + x


class BasicInfoResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.bn_1 = nn.BatchNorm2d(dim)
        self.act_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.bn_2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        residual = self.conv_1(x)
        residual = self.bn_1(residual)
        residual = self.act_1(residual)
        residual = self.conv_2(residual)
        residual = self.bn_2(residual)
        return x + residual


# model


class PredResnet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        depth=4,
        channels=3,
        self_condition=False,
        learned_sinusoidal_cond=False,
        learned_sinusoidal_dim=16,
    ):
        super().__init__()
        self.self_condition = self_condition
        # time embeddings

        time_dim = dim * 4

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

        self.depth = depth
        init_dim = init_dim or dim
        self.channels = channels
        self.init_conv_b = nn.Sequential(
            nn.Conv2d(channels * 3, init_dim, 3, 1, 1), nn.ReLU()
        )

        self.init_conv_n = nn.Sequential(
            nn.Conv2d(channels, init_dim, 3, 1, 1), nn.ReLU()
        )

        self.basic_info_resnets = nn.ModuleList()
        self.noise_resnets = nn.ModuleList()

        self.feats_merges = nn.ModuleList()

        for _ in range(depth):
            self.basic_info_resnets.append(BasicInfoResnetBlock(dim))
            self.noise_resnets.append(NoisePredResBlock(dim, time_emb_dim=time_dim))
            self.feats_merges.append(
                nn.Sequential(
                    nn.Conv2d(dim * 2, dim, 1), nn.BatchNorm2d(dim), nn.ReLU()
                )
            )
        self.final_feats_merge = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1), nn.BatchNorm2d(dim)
        )
        self.out_dim = out_dim
        self.final_conv = nn.Conv2d(dim, self.out_dim, 3, 1, 1)

    def forward(
        self,
        coarse_img_01,
        coarse_img_02,
        fine_img_01,
        noisy_fine_img_02,
        time,
        x_self_cond=None,
    ):
        x_b = torch.cat((coarse_img_01, coarse_img_02, fine_img_01), dim=1)

        x_n = noisy_fine_img_02

        x_b = self.init_conv_b(x_b)
        x_n = self.init_conv_n(x_n)

        r_b = x_b.clone()

        t = self.time_mlp(time)

        for i in range(self.depth):
            x_n = torch.cat((x_n, r_b), dim=1)
            x_n = self.feats_merges[i](x_n)
            x_n = self.noise_resnets[i](x_n, t)
            x_b = self.basic_info_resnets[i](x_b)

        x = torch.cat((x_b, x_n), dim=1)
        x = self.final_feats_merge(x)
        x = x + r_b
        x = self.final_conv(x)
        return x
