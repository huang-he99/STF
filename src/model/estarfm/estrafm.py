from numpy import pad
import torch
from torch import nn
import collections.abc
from itertools import repeat
from torch.nn.modules.utils import _pair, _quadruple
from typing import List, Tuple, Optional, overload, Dict, Any, Callable, Union
import numpy as np
from src.utils.constant import EPSILON


# Single prediction
class ESTARFM(nn.Module):
    def __init__(
        self,
        window_size,
        patch_size,
        num_classes,
        fine_img_uncertainty=0.002,
        coarse_img_uncertainty=0.002,
        relative_factor_spatial_distance=None,
        is_logistic=True,
        scale_factor=1.0,
    ):
        super().__init__()
        self.window_size = _pair(window_size)
        self.patch_size = _pair(patch_size)
        self.num_classes = num_classes
        self.window_unflod = nn.Unfold(self.window_size)
        self.window_mid_index = (
            self.window_size[1] * self.window_size[0] // 2 + self.window_size[1] // 2
        )
        self.spectral_uncertainty = (
            fine_img_uncertainty**2 + coarse_img_uncertainty**2
        ) ** 0.5
        self.temporal_uncertainty = (
            coarse_img_uncertainty**2 + coarse_img_uncertainty**2
        ) ** 0.5
        self.relative_factor_spatial_distance = (
            window_size // 2
            if relative_factor_spatial_distance is None
            else relative_factor_spatial_distance
        )
        self.is_logistic = is_logistic
        self.scale_factor = scale_factor
        self.nan = nn.Parameter(
            torch.Tensor((np.nan,)),
            requires_grad=False,
        )
        self.spatial_difference = nn.Parameter(
            self._spatial_difference(),
            requires_grad=False,
        )

    def _spatial_difference(self):
        pos_x, pos_y = torch.meshgrid(
            torch.arange(self.window_size[0]),
            torch.arange(self.window_size[1]),
            indexing='ij',
        )

        spatial_dis = (
            (
                (pos_x - self.window_size[0] // 2) ** 2
                + (pos_y - self.window_size[1] // 2) ** 2
            )
            .float()
            .flatten()
        )

        spatial_difference = spatial_dis / self.relative_factor_spatial_distance + 1
        return spatial_difference

    def forward(
        self,
        coarse_img_01,
        coarse_img_02,
        coarse_img_03,
        fine_img_01,
        fine_img_03,
    ):
        b, c, h, w = coarse_img_01.shape

        coarse_img_01_cube = self.get_cube(coarse_img_01, c)
        coarse_img_02_cube = self.get_cube(coarse_img_02, c)
        coarse_img_03_cube = self.get_cube(coarse_img_03, c)
        fine_img_01_cube = self.get_cube(fine_img_01, c)
        fine_img_03_cube = self.get_cube(fine_img_03, c)

        del coarse_img_01, coarse_img_02, coarse_img_03, fine_img_01, fine_img_03
        torch.cuda.empty_cache()

        pred_fine_cube = self._estarfm(
            coarse_img_01_cube,
            coarse_img_02_cube,
            coarse_img_03_cube,
            fine_img_01_cube,
            fine_img_03_cube,
        )
        pred_fine_cube = pred_fine_cube.transpose(-2, -1).reshape(
            b, c, *self.patch_size
        )
        return pred_fine_cube

    def get_cube(self, img_patch, c):
        img_cube = self.window_unflod(img_patch)
        b, c_muti_t_window_l, window_num = img_cube.shape
        img_cube = img_cube.permute(0, 2, 1).contiguous()
        assert c_muti_t_window_l % c == 0
        img_cube = img_cube.view(b, window_num, c, -1)
        return img_cube

    def _estarfm(
        self,
        coarse_img_01_cube,
        coarse_img_02_cube,
        coarse_img_03_cube,
        fine_img_01_cube,
        fine_img_03_cube,
    ):
        similar_neighbor_pixels_1 = self.similar_neighbor_pixel_selection(
            fine_img_01_cube
        )
        similar_neighbor_pixels_3 = self.similar_neighbor_pixel_selection(
            fine_img_03_cube
        )
        similar_neighbor_pixels = similar_neighbor_pixels_1 * similar_neighbor_pixels_3

        # Combined Weighting Function:
        spectral_correlation_coefficient = self.cal_spectral_correlation_coefficient(
            coarse_img_01_cube,
            coarse_img_03_cube,
            fine_img_01_cube,
            fine_img_03_cube,
            similar_neighbor_pixels,
        )

        weight = (
            (1.0 - spectral_correlation_coefficient)
            * self.spatial_difference
            * similar_neighbor_pixels
        )
        weight = 1.0 / (weight + EPSILON)

        # Sample Filtering
        filtered_pixels = self.sample_filtering(
            spectral_difference, temporal_difference
        )
        del spectral_difference, temporal_difference
        torch.cuda.empty_cache()
        # Spectrally Similar Neighbor Pixels:
        similar_neighbor_pixels = self.spectrally_similar_neighbor_pixels(
            prior_fine_cube
        )
        mask = similar_neighbor_pixels * filtered_pixels
        masked_weight = weight * mask
        norm_weight = masked_weight / (
            torch.sum(masked_weight, dim=(-2, -1), keepdim=True) + EPSILON
        )
        del masked_weight, mask, similar_neighbor_pixels, filtered_pixels
        torch.cuda.empty_cache()
        pred_fine_cube = pred_coarse_cube + prior_fine_cube - prior_coarse_cube
        del prior_coarse_cube, prior_fine_cube, pred_coarse_cube
        torch.cuda.empty_cache()
        pred_fine_cube = torch.sum(norm_weight * pred_fine_cube, dim=(-2, -1))
        # torch.einsum("bnctl,bnctl->bnc", norm_weight, pred_fine_cube)
        return pred_fine_cube

    def similar_neighbor_pixel_selection(self, fine_img_cube):
        fine_img_cube = torch.where(fine_img_cube < EPSILON, self.nan, fine_img_cube)
        std = torch.sqrt(
            torch.nanmean(fine_img_cube**2, keepdim=True)
            - torch.nanmean(fine_img_cube, keepdim=True) ** 2
        )
        threshold = 2 * std / self.num_classes
        center_difference = torch.abs(
            fine_img_cube - fine_img_cube[:, self.window_mid_index].unsqueeze(1)
        )
        similar_neighbor_pixels = torch.where(center_difference < threshold, 1, 0)
        return similar_neighbor_pixels

    def cal_spectral_correlation_coefficient(
        self,
        coarse_img_01_cube,
        coarse_img_03_cube,
        fine_img_01_cube,
        fine_img_03_cube,
        similar_neighbor_pixels,
    ):
        coarse_img_01_cube = torch.where(
            similar_neighbor_pixels == 1, coarse_img_01_cube, self.nan
        )
        coarse_img_03_cube = torch.where(
            similar_neighbor_pixels == 1, coarse_img_03_cube, self.nan
        )
        fine_img_01_cube = torch.where(
            similar_neighbor_pixels == 1, fine_img_01_cube, self.nan
        )
        fine_img_03_cube = torch.where(
            similar_neighbor_pixels == 1, fine_img_03_cube, self.nan
        )

        fine_img = torch.cat([fine_img_01_cube, fine_img_03_cube], dim=1)
        coarse_img = torch.cat([coarse_img_01_cube, coarse_img_03_cube], dim=1)

        fine_img_mean = torch.nanmean(fine_img, dim=(1, 2), keepdim=True)
        coarse_img_mean = torch.nanmean(coarse_img, dim=(1, 2), keepdim=True)

        fine_img_std = torch.sqrt(
            torch.nanmean(fine_img**2, dim=(1, 2), keepdim=True)
            - torch.nanmean(fine_img, dim=(1, 2), keepdim=True) ** 2
        )

        coarse_img_std = torch.sqrt(
            torch.nanmean(coarse_img**2, dim=(1, 2), keepdim=True)
            - torch.nanmean(coarse_img, dim=(1, 2), keepdim=True) ** 2
        )

        cor = torch.nanmean(fine_img * coarse_img, dim=(1, 2), keepdim=True) - (
            fine_img_mean * coarse_img_mean
        )

        spectral_correlation_coefficient = cor / (
            fine_img_std * coarse_img_std + EPSILON
        )

        return spectral_correlation_coefficient

    def combined_weighting_function(
        self, spectral_difference, temporal_difference, spatial_difference
    ):
        if self.is_logistic:
            weight = (
                torch.log(spectral_difference * self.scale_factor + 1)
                * torch.log(temporal_difference * self.scale_factor + 1)
                * spatial_difference
                + EPSILON
            )
        else:
            weight = (
                spectral_difference * temporal_difference * spatial_difference + EPSILON
            )
        return 1 / weight

    def sample_filtering(self, spectral_difference, temporal_difference):
        max_spectral_difference = torch.max(
            spectral_difference[:, :, :, :, self.window_mid_index].unsqueeze(-1),
            dim=-2,
            keepdim=True,
        )[0]
        max_temporal_difference = torch.max(
            temporal_difference[:, :, :, :, self.window_mid_index].unsqueeze(-1),
            dim=-2,
            keepdim=True,
        )[0]

        spectral_filter = torch.where(
            spectral_difference < max_spectral_difference + self.spectral_uncertainty,
            1,
            0,
        )
        temporal_filter = torch.where(
            temporal_difference < max_temporal_difference + self.temporal_uncertainty,
            1,
            0,
        )
        filter = spectral_filter * temporal_filter
        return filter

    def spectrally_similar_neighbor_pixels(self, prior_fine_cube):
        prior_fine_cube = torch.where(
            prior_fine_cube < EPSILON, self.nan, prior_fine_cube
        )
        std = torch.sqrt(
            torch.nanmean(prior_fine_cube**2, keepdim=True)
            - torch.nanmean(prior_fine_cube, keepdim=True) ** 2
        )
        threshold = 2 * std / self.num_classes
        center_difference = torch.abs(
            prior_fine_cube
            - prior_fine_cube[:, :, :, :, self.window_mid_index].unsqueeze(-1)
        )
        similar_neighbor_pixels = torch.where(center_difference < threshold, 1, 0)
        return similar_neighbor_pixels


# python -m src.model.starfm.strafm
if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    patch_size = 120
    window_size = 31
    num_classes = 5
    virtual_patch_size = patch_size + window_size - 1
    prior_coarse_img_patch = torch.randn(
        2, 6, 2, virtual_patch_size, virtual_patch_size
    ).to('cuda')
    prior_fine_img_patch = torch.randn(
        2, 6, 2, virtual_patch_size, virtual_patch_size
    ).to('cuda')
    pred_coarse_img_patch = torch.randn(
        2, 6, 1, virtual_patch_size, virtual_patch_size
    ).to('cuda')
    starfm = ESTARFM(
        window_size=window_size,
        patch_size=patch_size,
        num_classes=num_classes,
        fine_img_uncertainty=0.01,
        coarse_img_uncertainty=0.01,
    ).to('cuda')
    starfm.eval()
    with torch.no_grad():
        output = starfm(
            prior_coarse_img_patch, prior_fine_img_patch, pred_coarse_img_patch
        )
