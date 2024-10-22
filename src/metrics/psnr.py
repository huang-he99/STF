# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn
import torch


class PeakSignalNoiseRatio(nn.Module):
    """Peak Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Metrics:
        - PSNR (float): Peak Signal-to-Noise Ratio
    """

    __name__ = 'PSNR'

    def __init__(self, max_value=255):
        super().__init__()
        self.max_value = max_value

    def forward(self, gt, pred):
        """Process an image.

        Args:
            gt (Torch | np.ndarray): GT image.
            pred (Torch | np.ndarray): Pred image.
            mask (Torch | np.ndarray): Mask of evaluation.
        Returns:
            np.ndarray: PSNR result.
        """
        mse_value = ((gt - pred) ** 2).mean()
        if mse_value == 0:
            result = torch.tensor(torch.inf)
        else:
            result = 20.0 * torch.log10(self.max_value / torch.sqrt(mse_value))

        return result


PSNR = PeakSignalNoiseRatio
