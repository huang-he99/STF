# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn
import torch
from typing_extensions import Literal


class SpectralAngleMapper(nn.Module):
    """Peak Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Metrics:
        - PSNR (float): Peak Signal-to-Noise Ratio
    """

    __name__ = 'SAM'

    def __init__(self, unit: Literal['degree', 'rad'] = 'rad'):
        super().__init__()
        self.output_unit = unit

    def forward(self, gt, pred):
        """Process an image.

        Args:
            gt (Torch | np.ndarray): GT image.
            pred (Torch | np.ndarray): Pred image.
            mask (Torch | np.ndarray): Mask of evaluation.
        Returns:
            np.ndarray: PSNR result.
        """
        # inner_product = torch.einsum('bchw,bchw->bhw', gt, pred)
        # norm_gt = torch.einsum('bchw,bchw->bhw', gt, gt).sqrt()
        # norm_pred = torch.einsum('bchw,bchw->bhw', pred, pred).sqrt()
        inner_product = torch.sum(gt * pred, dim=1)
        norm_gt = torch.sum(gt * gt, dim=1).sqrt()
        norm_pred = torch.sum(pred * pred, dim=1).sqrt()
        pixel_cos = inner_product / (norm_gt * norm_pred)
        pixel_angle_rad = torch.acos(pixel_cos)
        if self.output_unit == 'rad':
            sam_rad = torch.nanmean(pixel_angle_rad)
            return sam_rad
        else:
            pixel_angle_degree = torch.rad2deg(pixel_angle_rad)
            sam_degree = torch.nanmean(pixel_angle_degree)
            return sam_degree


SAM = SpectralAngleMapper

# python -m src.metrics.sam
if __name__ == '__main__':
    sam = SAM()
    preds = torch.rand([16, 3, 16, 16], generator=torch.manual_seed(42))
    target = torch.rand([16, 3, 16, 16], generator=torch.manual_seed(123))
    print(sam(preds, target))
    # tensor(0.5943)
