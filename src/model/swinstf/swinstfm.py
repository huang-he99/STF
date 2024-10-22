import torch.nn as nn

from .fem import Down
from .mfm import FineUp


class SwinSTFM(nn.Module):
    def __init__(self):
        super(SwinSTFM, self).__init__()
        self.coarse_down = Down()
        self.fine_down = Down()
        self.fine_up = FineUp()

    def forward(self, c0, c1, f0):
        coarse_fea0 = self.coarse_down(c0)
        coarse_fea1 = self.coarse_down(c1)
        fine_fea = self.fine_down(f0)
        diff_fea1 = []
        for i in range(5):
            diff_fea1.append(coarse_fea1[i] - coarse_fea0[i])

        diff_fea0 = []
        for i in range(5):
            diff_fea0.append(fine_fea[i] - coarse_fea0[i])

        output_fine = self.fine_up(coarse_fea0, diff_fea1, fine_fea, coarse_fea1)

        # output_fine_description = [
        #     'PixleLoss_preds',
        #     'MSSSIMLoss_preds',
        #     'RMSE_preds',
        #     'PSNR_preds',
        #     'ERGAS_preds',
        #     'SAM_preds',
        #     'SSIM_preds',
        # ]
        return output_fine
