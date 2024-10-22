import torch
import torch.nn.functional as F
from torch import nn
from .cnnblock import ResidualBlock, UpsampleBLock


class STFGANGenerator(nn.Module):
    def __init__(self, img_channel_num):
        super(STFGANGenerator, self).__init__()
        '''
        # ------------------------------------------------
        # Coarse images feature extraction branch CIFEB
        # ------------------------------------------------
        '''
        self.CIFEB_RBs_head = nn.Sequential(
            nn.Conv2d(img_channel_num * 3, 64, kernel_size=9, padding=4), nn.ReLU()
        )
        self.CIFEB_RBs = nn.ModuleList()
        for _ in range(16):
            self.CIFEB_RBs.append(ResidualBlock(64))
        self.CIFEB_RBs_tail = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64)
        )

        self.CIFEB_upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
        )

        '''
        # ------------------------------------------------
        # Fine images feature extraction branch FIFEB
        # ------------------------------------------------
        '''
        self.FIFEB_RBs_head = nn.Sequential(
            nn.Conv2d(img_channel_num * 2, 64, kernel_size=9, padding=4), nn.ReLU()
        )
        self.FIFEB_RBs = nn.ModuleList()
        for _ in range(8):
            self.FIFEB_RBs.append(ResidualBlock(64))
        self.FIFEB_RBs_tail = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64)
        )

        '''
        # ----------------------------------------
        # Two branches fusion TBF
        # ----------------------------------------
        '''
        self.TBF_conv = nn.Conv2d(128, img_channel_num, kernel_size=9, padding='same')

    def forward(
        self, coarse_img_01, coarse_img_02, coarse_img_03, fine_img_01, fine_img_03
    ):
        '''
        # ------------------------------------------------
        # Coarse images feature extraction branch CIFEB
        # ------------------------------------------------
        '''
        coarse_img_concat = torch.cat((coarse_img_01, coarse_img_02, coarse_img_03), 1)
        coarse_feats = self.CIFEB_RBs_head(coarse_img_concat)
        coarse_feats_residual = coarse_feats
        for i in range(16):
            coarse_feats_residual = self.CIFEB_RBs[i](coarse_feats_residual)
        coarse_feats_residual = self.CIFEB_RBs_tail(coarse_feats_residual)
        coarse_feats = coarse_feats + coarse_feats_residual
        upsample_coarse_feats = self.CIFEB_upsample(coarse_feats)
        '''
        # ------------------------------------------------
        # Fine images feature extraction branch FIFEB
        # ------------------------------------------------
        '''
        fine_img_concat = torch.cat((fine_img_01, fine_img_03), 1)
        fine_feats = self.FIFEB_RBs_head(fine_img_concat)
        fine_feats_residual = fine_feats
        for i in range(8):
            fine_feats_residual = self.FIFEB_RBs[i](fine_feats_residual)
        fine_feats_residual = self.FIFEB_RBs_tail(fine_feats_residual)
        fine_feats = fine_feats + fine_feats_residual
        '''
        # ----------------------------------------
        # Two branches fusion TBF
        # ----------------------------------------
        '''
        fusion_feats = torch.cat((upsample_coarse_feats, fine_feats), 1)
        x = torch.tanh(self.TBF_conv(fusion_feats))
        return x
