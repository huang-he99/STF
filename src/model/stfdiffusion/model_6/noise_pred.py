import math
from functools import partial


import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce


# helpers functions


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


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


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


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


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()

        self.proj_1 = WeightStandardizedConv2d(dim, dim_out, 3, 1, 1)
        self.norm_1 = nn.GroupNorm(8, dim_out)
        self.act_1 = nn.SiLU()

        self.proj_2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm_2 = nn.GroupNorm(8, dim_out)
        self.act_2 = nn.SiLU()

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.proj_1(x)
        h = self.norm_1(h)
        # scale, shift = scale_shift
        # h = h * (scale + 1) + shift
        h = self.act_1(h)

        h = self.proj_2(h)
        h = self.norm_2(h)
        h = self.act_2(h)
        return h + self.res_conv(x)


class ResnetBlockwithT(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))

        self.proj_1 = WeightStandardizedConv2d(dim, dim_out, 3, 1, 1)
        self.norm_1 = nn.GroupNorm(8, dim_out)
        self.act_1 = nn.SiLU()

        self.proj_2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm_2 = nn.GroupNorm(8, dim_out)
        self.act_2 = nn.SiLU()

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        time_emb = self.mlp(time_emb)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        scale_shift = time_emb.chunk(2, dim=1)

        h = self.proj_1(x)
        h = self.norm_1(h)
        scale, shift = scale_shift
        h = h * (scale + 1) + shift

        h = self.act_1(h)

        h = self.proj_2(h)
        h = self.norm_2(h)
        h = self.act_2(h)

        return h + self.res_conv(x)


# model


class NoisePredNet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4),
        channels=6,
        self_condition=False,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        learned_sinusoidal_dim=16,
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
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

        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)

        self.init_conv_fine = nn.Conv2d(channels, init_dim, 3, 1, 1)
        self.init_conv_coarse = nn.Conv2d(channels * 2, init_dim, 3, 1, 1)
        self.init_conv_noisy = nn.Conv2d(channels, init_dim, 3, 1, 1)

        self.init_conv_clean = nn.Conv2d(init_dim * 2, init_dim, 1)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.clean_downs = nn.ModuleList([])
        self.noisy_downs = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)

            clean_down = nn.ModuleList(
                [
                    ResnetBlock(dim_in, dim_in),
                    Downsample(dim_in, dim_out)
                    if not is_last
                    else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                ]
            )
            noisy_down = nn.ModuleList(
                [
                    ResnetBlockwithT(dim_in, dim_in, time_emb_dim=time_dim),
                    Downsample(dim_in, dim_out)
                    if not is_last
                    else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                ]
            )

            self.clean_downs.append(clean_down)
            self.noisy_downs.append(noisy_down)

        mid_dim = dims[-1]
        self.mid_block_clean = ResnetBlock(mid_dim, mid_dim)
        self.mid_block_noisy = ResnetBlockwithT(mid_dim, mid_dim, time_emb_dim=time_dim)

        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            up = nn.ModuleList(
                [
                    ResnetBlockwithT(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                    Upsample(dim_out, dim_in)
                    if not is_last
                    else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                ]
            )

            self.ups.append(up)

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = ResnetBlockwithT(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(
        self,
        coarse_img_01,
        coarse_img_02,
        fine_img_01,
        noisy_fine_img_02,
        time,
        x_self_cond=None,
    ):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        # fine_img_diff = noisy_fine_img_02 - fine_img_01
        # coarse_img_diff = coarse_img_02 - coarse_img_01
        # x = torch.cat((fine_img_diff, coarse_img_diff), dim=1)
        # x = torch.cat(
        #     (coarse_img_diff, noisy_fine_img_diff),
        #     dim=1,
        # )

        fine_img = fine_img_01
        coarse_img = torch.cat((coarse_img_01, coarse_img_02), dim=1)

        x_fine = self.init_conv_fine(fine_img)
        x_coarse = self.init_conv_coarse(coarse_img)
        x_noisy = self.init_conv_noisy(noisy_fine_img_02)

        x_clean = torch.cat((x_fine, x_coarse), dim=1)

        x_clean = self.init_conv_clean(x_clean)

        r = x_noisy - x_clean

        t = self.time_mlp(time)

        h = []

        down_num = len(self.clean_downs)

        for down_idx in range(down_num):
            clean_down_res, clean_downsample = self.clean_downs[down_idx]
            noisy_down_res, noisy_downsample = self.noisy_downs[down_idx]
            x_clean = clean_down_res(x_clean)
            x_noisy = noisy_down_res(x_noisy, t)
            noise = x_noisy - x_clean
            h.append(noise)
            x_clean = clean_downsample(x_clean)
            x_noisy = noisy_downsample(x_noisy)

        x_clean = self.mid_block_clean(x_clean)
        x_noisy = self.mid_block_noisy(x_noisy, t)

        noise = x_noisy - x_clean

        up_num = len(self.ups)

        for up_idx in range(up_num):
            up_res, upsample = self.ups[up_idx]
            noise = torch.cat((noise, h.pop()), dim=1)
            noise = up_res(noise, t)
            noise = upsample(noise)

        noise = torch.cat((noise, r), dim=1)

        noise = self.final_res_block(noise, t)
        return self.final_conv(noise)
