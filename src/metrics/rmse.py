# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn
import torch


class RootMeanSquareError(nn.Module):
    """Peak Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Metrics:
        - PSNR (float): Peak Signal-to-Noise Ratio
    """

    __name__ = 'RMSE'

    def __init__(self,is_reduce_channel=True):
        super().__init__()
        self.is_reduce_channel = is_reduce_channel

    def forward(self, gt, pred):
        """Process an image.

        Args:
            gt (Torch | np.ndarray): GT image.
            pred (Torch | np.ndarray): Pred image.
            mask (Torch | np.ndarray): Mask of evaluation.
        Returns:
            np.ndarray: PSNR result.
        """
        if self.is_reduce_channel:
            mse_value = ((gt - pred) ** 2).mean()
        else:
            mse_value = ((gt - pred) ** 2).mean(dim=(0, 2, 3))
 
        result = torch.sqrt(mse_value)
        return result


RMSE = RootMeanSquareError
