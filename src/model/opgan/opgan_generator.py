import torch
import torch.nn.functional as F
from torch import nn
from functools import partial
from .cnnblock import ResidualBlock

# from .cnnblock import ResidualBlock, UpsampleBLock
from typing import Optional


# class ResidualBlock(nn.Module):
#     def __init__(
#         self,
#         in_planes: int,
#         planes: int,
#         norm: Optional[nn.Module] = None,
#         act: nn.Module = partial(nn.ReLU, inplace=True),
#         res_ratio: int = 1,
#     ) -> None:
#         super().__init__()
#         self.conv_1 = nn.Conv2d(in_planes, planes, 3, 1, 1)
#         self.norm_1 = norm(planes) if norm is not None else nn.Identity()
#         self.act = act()
#         self.conv_2 = nn.Conv2d(planes, planes, 3, 1, 1)
#         self.norm_2 = norm(planes) if norm is not None else nn.Identity()
#         self.res_ratio = res_ratio

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out = self.conv_1(x)
#         out = self.norm_1(out)
#         out = self.act(out)
#         out = self.conv_2(out)
#         out = self.norm_2(out)
#         out = x * self.res_ratio + out
#         return out


# class StackedResidualBlock(nn.Module):
#     def __init__(
#         self,
#         in_planes: int,
#         planes: int,
#         residual_block_num: int,
#         norm: Optional[nn.Module] = None,
#         act: nn.Module = partial(nn.ReLU, inplace=True),
#     ):
#         super().__init__()
#         self.blocks = nn.ModuleList()
#         for _ in range(residual_block_num):
#             self.blocks.append(ResidualBlock(in_planes, planes, norm, act))
#             in_planes = planes

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out = x
#         for block in self.blocks:
#             out = block(out)
#         return out


# class FeatureExtraction(nn.Module):
#     def __init__(
#         self,
#         in_channels: int = 6,
#         planes: int = 64,
#         residual_block_num: int = 8,
#         act: nn.Module = partial(nn.ReLU, inplace=True),
#         norm: nn.Module = nn.BatchNorm2d,
#     ):
#         super().__init__()
#         self.head = nn.Sequential(nn.Conv2d(in_channels, planes, 3, 1, 1), act())
#         self.stacked_residual_block = StackedResidualBlock(
#             in_planes=planes,
#             planes=planes,
#             residual_block_num=residual_block_num,
#             norm=norm,
#             act=act,
#         )
#         self.tail = nn.Sequential(nn.Conv2d(planes, planes, 3, 1, 1), norm(planes))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.head(x)
#         res = self.stacked_residual_block(x)
#         res = self.tail(res)
#         x = x + res
#         return x


# class SensorDifferenceExtraction(FeatureExtraction):
#     def __init__(
#         self,
#         in_channels: int = 6,
#         planes: int = 64,
#         residual_block_num: int = 4,
#         act: nn.Module = partial(nn.ReLU, inplace=True),
#         norm: nn.Module = nn.BatchNorm2d,
#     ):
#         super().__init__(
#             in_channels=in_channels,
#             planes=planes,
#             residual_block_num=residual_block_num,
#             act=act,
#             norm=norm,
#         )


# class TemporalChangeCapture(FeatureExtraction):
#     def __init__(
#         self,
#         in_channels: int = 6,
#         planes: int = 64,
#         residual_block_num: int = 8,
#         act: nn.Module = partial(nn.ReLU, inplace=True),
#         norm: nn.Module = nn.BatchNorm2d,
#     ):
#         super().__init__(
#             in_channels=in_channels,
#             planes=planes,
#             residual_block_num=residual_block_num,
#             act=act,
#             norm=norm,
#         )


# class BaseInformationLearning(FeatureExtraction):
#     def __init__(
#         self,
#         in_channels: int = 6,
#         planes: int = 64,
#         residual_block_num: int = 2,
#         act: nn.Module = partial(nn.ReLU, inplace=True),
#         norm: nn.Module = nn.BatchNorm2d,
#     ):
#         super().__init__(
#             in_channels=in_channels,
#             planes=planes,
#             residual_block_num=residual_block_num,
#             act=act,
#             norm=norm,
#         )


class OPGANGenerator(nn.Module):
    def __init__(self, img_channel_num: int = 6):
        super().__init__()
        '''
        # ------------------------------------------------
        # Temporal Change Capture branch TCCB
        # ------------------------------------------------
        '''
        self.TCCB_RBs_head = nn.Sequential(
            nn.Conv2d(img_channel_num, 64, kernel_size=9, padding=4), nn.ReLU()
        )

        self.TCCB_RBs = nn.ModuleList()
        for _ in range(8):
            self.TCCB_RBs.append(ResidualBlock(64))
        self.TCCB_RBs_tail = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64)
        )

        '''
        # ------------------------------------------------
        # Sensor Difference Extraction branch SDEB
        # ------------------------------------------------
        '''
        self.SDEB_RBs_head = nn.Sequential(
            nn.Conv2d(img_channel_num, 64, kernel_size=9, padding=4), nn.ReLU()
        )

        self.SDEB_RBs = nn.ModuleList()
        for _ in range(4):
            self.SDEB_RBs.append(ResidualBlock(64))
        self.SDEB_RBs_tail = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64)
        )

        '''
        # ------------------------------------------------
        # Base Information Learning branch BILB
        # ------------------------------------------------
        '''
        self.BILB_RBs_head = nn.Sequential(
            nn.Conv2d(img_channel_num, 64, kernel_size=9, padding=4), nn.ReLU()
        )

        self.BILB_RBs = nn.ModuleList()
        for _ in range(2):
            self.BILB_RBs.append(ResidualBlock(64))
        self.BILB_RBs_tail = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64)
        )

        self.tail = nn.Sequential(
            nn.Conv2d(64, img_channel_num, 9, 1, padding='same'), nn.Tanh()
        )

        self.temporal_change_tail = nn.Sequential(
            nn.Conv2d(64, img_channel_num, 9, 1, padding='same'), nn.Tanh()
        )

    def forward(
        self,
        coarse_img_01: torch.tensor,
        coarse_img_02: torch.tensor,
        fine_img_01: torch.tensor,
    ) -> torch.tensor:
        '''
        # ------------------------------------------------
        # Temporal Change Capture branch TCCB
        # ------------------------------------------------
        '''
        coarse_temporal_change_img = coarse_img_02 - coarse_img_01
        coarse_temporal_change_feats = self.TCCB_RBs_head(coarse_temporal_change_img)
        coarse_temporal_change_feats_residuals = coarse_temporal_change_feats
        for i in range(8):
            coarse_temporal_change_feats_residuals = self.TCCB_RBs[i](
                coarse_temporal_change_feats_residuals
            )
        coarse_temporal_change_feats_residuals = self.TCCB_RBs_tail(
            coarse_temporal_change_feats_residuals
        )
        coarse_temporal_change_feats = (
            coarse_temporal_change_feats + coarse_temporal_change_feats_residuals
        )

        '''
        # ------------------------------------------------
        # Sensor Difference Extraction branch SDEB
        # ------------------------------------------------
        '''
        sensor_difference_img = fine_img_01 - coarse_img_01
        sensor_difference_feats = self.SDEB_RBs_head(sensor_difference_img)
        sensor_difference_feats_residuals = sensor_difference_feats
        for i in range(4):
            sensor_difference_feats_residuals = self.SDEB_RBs[i](
                sensor_difference_feats_residuals
            )
        sensor_difference_feats_residuals = self.SDEB_RBs_tail(
            sensor_difference_feats_residuals
        )
        sensor_difference_feats = (
            sensor_difference_feats + sensor_difference_feats_residuals
        )

        '''
        # ------------------------------------------------
        # Base Information Learning branch BILB
        # ------------------------------------------------
        '''
        base_information_img = fine_img_01
        base_information_feats = self.BILB_RBs_head(base_information_img)
        base_information_feats_residuals = base_information_feats
        for i in range(2):
            base_information_feats_residuals = self.BILB_RBs[i](
                base_information_feats_residuals
            )
        base_information_feats_residuals = self.BILB_RBs_tail(
            base_information_feats_residuals
        )
        base_information_feats = (
            base_information_feats + base_information_feats_residuals
        )

        '''
        # ------------------------------------------------
        # fine_temporal_change
        # ------------------------------------------------
        '''
        fine_temporal_change_feats = (
            coarse_temporal_change_feats + sensor_difference_feats
        )
        fine_temporal_change_img = self.temporal_change_tail(fine_temporal_change_feats)

        '''
        # ------------------------------------------------
        # fine_img
        # ------------------------------------------------
        '''

        fine_feats = (
            coarse_temporal_change_feats
            + sensor_difference_feats
            + base_information_feats
        )
        fine_img = self.tail(fine_feats)

        return fine_img, fine_temporal_change_img
