# Correlation Coefficient
import statistics
from torch import nn
import torch
import cv2
import torch.nn.functional as F

epsilon = 1e-10


# class UniversalImageQualityIndex(nn.Module):
#     """Peak Signal-to-Noise Ratio.

#     Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

#     Metrics:
#         - PSNR (float): Peak Signal-to-Noise Ratio
#     """

#     __name__ = 'UIQI'

#     def __init__(
#         self,
#         gussian_kernel_size=11,
#         gussian_sigma=1.5,
#         data_range=255,
#         K1=0.01,
#         K2=0.03,
#         is_reduce_channel=True,
#     ):
#         super().__init__()
#         self.gussian_kernel = self._get_gussian_kernel(
#             gussian_kernel_size, gussian_sigma
#         )
#         self.C1 = (K1 * data_range) ** 2
#         self.C2 = (K2 * data_range) ** 2
#         self.unfold = nn.Unfold(gussian_kernel_size)
#         self.gussian_kernel_size = gussian_kernel_size
#         self.is_reduce_channel = is_reduce_channel

#     def _get_gussian_kernel(self, gussian_kernel_size, gussian_sigma):
#         d_kernel = cv2.getGaussianKernel(gussian_kernel_size, gussian_sigma)
#         _2d_kernel = d_kernel @ d_kernel.T
#         kernel = torch.FloatTensor(_2d_kernel).unsqueeze(0).unsqueeze(0)
#         kernel = nn.Parameter(kernel, requires_grad=False)
#         return kernel

#     def forward(self, gt, pred):
#         """Process an image.

#         Args:
#             BCHW format.
#             gt (Torch | np.ndarray): GT image.
#             pred (Torch | np.ndarray): Pred image.
#             mask (Torch | np.ndarray): Mask of evaluation.
#         Returns:
#             np.ndarray: PSNR result.
#         """
#         assert gt.shape == pred.shape
#         b, c, h, w = gt.shape
#         kernel = self.gussian_kernel.repeat_interleave(repeats=c, dim=0)

#         cube = torch.cat((gt, pred, gt * gt, pred * pred, gt * pred), dim=0)

#         statistic_cube = F.conv2d(cube, weight=kernel, groups=c)

#         mu_gt_sq = statistic_cube[:b] ** 2
#         mu_pred_sq = statistic_cube[b : 2 * b] ** 2
#         mu_gt_mul_pred = statistic_cube[:b] * statistic_cube[b : 2 * b]

#         sigma_gt_sq = statistic_cube[2 * b : 3 * b] - mu_gt_sq
#         sigma_pred_sq = statistic_cube[3 * b : 4 * b] - mu_pred_sq

#         cor_gt_pred = statistic_cube[4 * b :] - mu_gt_mul_pred


#         uiqi_map = (
#             4
#             * cor_gt_pred
#             * mu_gt_mul_pred
#             / (
#                 (sigma_gt_sq + sigma_pred_sq + epsilon)
#                 * (mu_gt_sq + mu_pred_sq + epsilon)
#             )
#         )
#         if self.is_reduce_channel:
#             return torch.mean(uiqi_map)
#         else:
#             return torch.mean(uiqi_map, dim=(0, -1))
class UniversalImageQualityIndex(nn.Module):
    """Peak Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Metrics:
        - PSNR (float): Peak Signal-to-Noise Ratio
    """

    __name__ = 'UIQI'

    def __init__(
        self,
        # gussian_kernel_size=11,
        # gussian_sigma=1.5,
        is_reduce_channel=True,
    ):
        super().__init__()
        # self.gussian_kernel = self._get_gussian_kernel(
        # gussian_kernel_size, gussian_sigma
        # )

        self.is_reduce_channel = is_reduce_channel

    # def _get_gussian_kernel(self, gussian_kernel_size, gussian_sigma):
    #     d_kernel = cv2.getGaussianKernel(gussian_kernel_size, gussian_sigma)
    #     _2d_kernel = d_kernel @ d_kernel.T
    #     kernel = torch.FloatTensor(_2d_kernel).unsqueeze(0).unsqueeze(0)
    #     kernel = nn.Parameter(kernel, requires_grad=False)
    #     return kernel

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
        # assert gt.shape == pred.shape
        # b, c, h, w = gt.shape
        # # kernel = self.gussian_kernel.repeat_interleave(repeats=c, dim=0)

        # cube = torch.cat((gt, pred, gt * gt, pred * pred, gt * pred), dim=0)
        # statistic_cube = cube

        # # statistic_cube = F.conv2d(cube, weight=kernel, groups=c)

        # mu_gt_sq = statistic_cube[:b] ** 2
        # mu_pred_sq = statistic_cube[b : 2 * b] ** 2
        # mu_gt_mul_pred = statistic_cube[:b] * statistic_cube[b : 2 * b]

        # sigma_gt_sq = statistic_cube[2 * b : 3 * b] - mu_gt_sq
        # sigma_pred_sq = statistic_cube[3 * b : 4 * b] - mu_pred_sq

        # cor_gt_pred = statistic_cube[4 * b :] - mu_gt_mul_pred

        # uiqi_map = (
        #     4
        #     * cor_gt_pred
        #     * mu_gt_mul_pred
        #     / (
        #         (sigma_gt_sq + sigma_pred_sq + epsilon)
        #         * (mu_gt_sq + mu_pred_sq + epsilon)
        #     )
        # )
        # if self.is_reduce_channel:
        #     return torch.mean(uiqi_map)
        # else:
        #     return torch.mean(uiqi_map, dim=(0, -1))

        assert gt.shape == pred.shape
        # b, c, h, w = gt.shape
        # kernel = self.gussian_kernel.repeat_interleave(repeats=c, dim=0)

        cube = torch.cat((gt, pred, gt * gt, pred * pred, gt * pred), dim=0)
        statistic_cube = cube

        # statistic_cube = F.conv2d(cube, weight=kernel, groups=c)

        mu_gt = torch.mean(gt, dim=(-2, -1), keepdim=True)
        mu_pred = torch.mean(pred, dim=(-2, -1), keepdim=True)

        mu_gt_sq = mu_gt**2
        mu_pred_sq = mu_pred**2

        mu_gt_mul_mu_pred = mu_gt * mu_pred

        sigma_gt_sq = torch.mean(gt**2, dim=(-2, -1), keepdim=True) - mu_gt_sq
        sigma_pred_sq = torch.mean(pred**2, dim=(-2, -1), keepdim=True) - mu_pred_sq

        cor_gt_pred = (
            torch.mean(gt * pred, dim=(-2, -1), keepdim=True) - mu_gt_mul_mu_pred
        )

        uiqi_map = (
            4
            * cor_gt_pred
            * mu_gt_mul_mu_pred
            / (
                (sigma_gt_sq + sigma_pred_sq + epsilon)
                * (mu_gt_sq + mu_pred_sq + epsilon)
            )
        )
        if self.is_reduce_channel:
            return torch.mean(uiqi_map)
        else:
            return torch.mean(uiqi_map, dim=(0, -2, -1))


UIQI = UniversalImageQualityIndex

# python -m src.metrics.uiqi
if __name__ == '__main__':
    preds = torch.rand([16, 16, 160, 160], generator=torch.manual_seed(42)).to('cpu')
    target = torch.rand([16, 16, 160, 160], generator=torch.manual_seed(40)).to('cpu')

    uiqi = UIQI(data_range=1.0)
    a = uiqi(target, preds)
    print(a)

    from torchmetrics import UniversalImageQualityIndex

    uqi = UniversalImageQualityIndex()
    b = uqi(preds, target)
    print(b)
