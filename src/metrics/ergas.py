# Copyright (c) OpenMMLab. All rights reserved.
from cv2 import sqrt
from torch import nn
import torch
from typing_extensions import Literal


class ErrorRelativeGlobalDimensionlessSynthesis(nn.Module):
    """Peak Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Metrics:
        - PSNR (float): Peak Signal-to-Noise Ratio
    """

    __name__ = 'ergas'

    def __init__(self, ratio=16):
        super().__init__()
        self.ratio = ratio

    def forward(self, gt, pred):
        """Process an image.

        Args:
            gt (Torch | np.ndarray): GT image.
            pred (Torch | np.ndarray): Pred image.
            mask (Torch | np.ndarray): Mask of evaluation.
        Returns:
            np.ndarray: PSNR result.
        """
        diff = gt - pred
        mse_per_channel = torch.mean(diff * diff, dim=(-2, -1))
        mu_gt_sq_per_channel = torch.mean(gt, dim=(-2, -1)).pow(2)
        ergas_batch = (
            100
            * self.ratio
            * torch.mean(mse_per_channel / mu_gt_sq_per_channel, dim=1).sqrt()
        )
        ergas = ergas_batch.mean()

        return ergas


ERGAS = ErrorRelativeGlobalDimensionlessSynthesis

# python -m src.metrics.ergas
if __name__ == '__main__':
    ergas = ERGAS()
    preds = torch.rand([16, 1, 16, 16], generator=torch.manual_seed(42))
    target = preds * 0.75
    print(ergas(target, preds))

    from torchmetrics import ErrorRelativeGlobalDimensionlessSynthesis

    ergas = ErrorRelativeGlobalDimensionlessSynthesis(16)
    print(ergas(preds, target))
    # tensor(0.5943)
