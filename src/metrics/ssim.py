# Copyright (c) OpenMMLab. All rights reserved.
import statistics
from torch import nn
import torch
import cv2
import torch.nn.functional as F


class StructuralSimilarity(nn.Module):
    """Peak Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Metrics:
        - PSNR (float): Peak Signal-to-Noise Ratio
    """

    __name__ = 'SSIM'

    def __init__(
        self,
        gussian_kernel_size=11,
        gussian_sigma=1.5,
        data_range=255,
        K1=0.01,
        K2=0.03,
        is_reduce_channel=True,
    ):
        super().__init__()
        self.gussian_kernel = self._get_gussian_kernel(
            gussian_kernel_size, gussian_sigma
        )
        self.C1 = (K1 * data_range) ** 2
        self.C2 = (K2 * data_range) ** 2
        self.unfold = nn.Unfold(gussian_kernel_size)
        self.gussian_kernel_size = gussian_kernel_size
        self.is_reduce_channel = is_reduce_channel

    def _get_gussian_kernel(self, gussian_kernel_size, gussian_sigma):
        d_kernel = cv2.getGaussianKernel(gussian_kernel_size, gussian_sigma)
        _2d_kernel = d_kernel @ d_kernel.T
        kernel = torch.FloatTensor(_2d_kernel).unsqueeze(0).unsqueeze(0)
        kernel = nn.Parameter(kernel, requires_grad=False)
        return kernel

    def forward(self, gt, pred):
        """Process an image.

        Args:
            BCHW format.
            gt (Torch | np.ndarray): GT image.
            pred (Torch | np.ndarray): Pred image.
            mask (Torch | np.ndarray): Mask of evaluation.
        Returns:
            np.ndarray: PSNR result.
        """
        assert gt.shape == pred.shape
        b, c, h, w = gt.shape

        # gt_unfold = self.unfold(gt).view(
        #     b, c, self.gussian_kernel_size, self.gussian_kernel_size, -1
        # )
        # pred_unfold = self.unfold(pred).view(
        #     b, c, self.gussian_kernel_size, self.gussian_kernel_size, -1
        # )

        # # E[X] = \inf xp(x)dx, D[X] = \inf (x-E[X])^2 p(x)dx, Cor[X, Y] = \inf \inf (x-E[x])(y-E[y])p(x,y)dxdy
        # # 这里积分域为 I = [-win_size/2, win_size/2] \times [-win_size/2, win_size/2] \cap (Z \times Z)
        # # E[X] = \sum\sum x(i,j)p(i,j) D[X] = \sum\sum (x(i,j)-E[X])^2 p(i,j), Cor[X,Y] = \sum\sum (x(i,j)-E[x])(y(i,j)-E[y])p(i,j)dxdy

        # mu_gt = torch.einsum('bcwhn,wh->bcn', gt_unfold, self.gussian_kernel)
        # mu_pred = torch.einsum('bcwhn,wh->bcn', pred_unfold, self.gussian_kernel)

        # var_gt = (
        #     torch.einsum('bcwhn,wh->bcn', gt_unfold * gt_unfold, self.gussian_kernel)
        #     - mu_gt**2
        # )
        # var_pred = (
        #     torch.einsum(
        #         'bcwhn,wh->bcn', pred_unfold * pred_unfold, self.gussian_kernel
        #     )
        #     - mu_pred**2
        # )
        # cor_gt_pred = (
        #     torch.einsum('bcwhn,wh->bcn', gt_unfold * pred_unfold, self.gussian_kernel)
        #     - mu_gt * mu_pred
        # )

        # cs_map = (2 * cor_gt_pred + self.C2) / (
        #     var_gt + var_pred + self.C2
        # )  # set alpha=beta=gamma=1
        # ssim_map = (
        #     (2 * mu_gt * mu_pred + self.C1)
        #     / (mu_gt * mu_gt + mu_pred * mu_pred + self.C1)
        # ) * cs_map

        # if self.is_reduce_channel:
        #     return torch.mean(ssim_map)
        # else:
        #     return torch.mean(ssim_map, dim=(0, -1))
        kernel = self.gussian_kernel.repeat_interleave(repeats=c, dim=0)

        cube = torch.cat((gt, pred, gt * gt, pred * pred, gt * pred), dim=0)

        statistic_cube = F.conv2d(cube, weight=kernel, groups=c)

        mu_gt_sq = statistic_cube[:b] ** 2
        mu_pred_sq = statistic_cube[b : 2 * b] ** 2
        mu_gt_mul_pred = statistic_cube[:b] * statistic_cube[b : 2 * b]

        sigma_gt_sq = statistic_cube[2 * b : 3 * b] - mu_gt_sq
        sigma_pred_sq = statistic_cube[3 * b : 4 * b] - mu_pred_sq

        cor_gt_pred = statistic_cube[4 * b :] - mu_gt_mul_pred

        cs_map = (2 * cor_gt_pred + self.C2) / (
            sigma_gt_sq + sigma_pred_sq + self.C2
        )  # set alpha=beta=gamma=1
        ssim_map = (
            (2 * mu_gt_mul_pred + self.C1) / (mu_gt_sq + mu_pred_sq + self.C1)
        ) * cs_map
        if self.is_reduce_channel:
            return torch.mean(ssim_map)
        else:
            return torch.mean(ssim_map, dim=(0, 2, 3))


SSIM = StructuralSimilarity

# python -m src.metrics.ssim
if __name__ == '__main__':
    preds = torch.rand([16, 3, 256, 256], generator=torch.manual_seed(42)).to('cpu')
    target = torch.rand([16, 3, 256, 256], generator=torch.manual_seed(123)).to('cpu')

    from torchmetrics import StructuralSimilarityIndexMeasure

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to('cpu')
    b = ssim(target, preds)
    print(ssim(target, preds))

    ssim = SSIM(data_range=1.0).to('cpu')
    a = ssim(target, preds)
    print(ssim(target, preds))

    ssim_cuda = StructuralSimilarityIndexMeasure(data_range=1.0).to('cuda')
    c = ssim_cuda(target.clone().cuda(), preds.clone().cuda())
    print(c)
    print(b - a)
