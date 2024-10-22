from torch import nn
import torch

# class OPGANDiscriminator(nn.Module):
#     def __init__(
#         self,
#         in_channels: int = 6,
#         dim: int = 64,
#         block_num: int = 7,
#         norm: Optional[nn.Module] = nn.BatchNorm2d,
#         act: Optional[nn.Module] = partial(
#             nn.LeakyReLU, in_place=True, negative_slope=0.2
#         ),
#     ):
#         super().__init__()
#         self.head = nn.Sequential(nn.Conv2d(in_channels, dim, 3, 1, 1), nn.LeakyReLU())
#         self.feature_extraction = nn.Sequential(
#             *(
#                 ConvNormActBlock(
#                     in_channels=dim * (2 ** (i // 2)),
#                     out_channels=dim * (2 ** ((i + 1)) // 2),
#                     kernel_size=3,
#                     stride=1 + ((i + 1)) % 2,
#                     padding=1,
#                     norm=norm,
#                     act=act,
#                 )
#                 for i in range(block_num)
#             )
#         )

#         self.classifier = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(512, 1024, 1),
#             act(),
#             nn.Conv2d(1024, 1, 1),
#         )

#     def forward(self, fine_or_fusion_img: torch.tensor) -> torch.tensor:
#         fine_or_fusion_feats = self.head(fine_or_fusion_img)
#         fine_or_fusion_feats = self.feature_extraction(fine_or_fusion_feats)
#         logits = self.classifier(fine_or_fusion_feats)
#         return logits


class OPGANDiscriminator(nn.Module):
    def __init__(
        self,
        img_channel_num: int = 6,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channel_num, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1),
        )

    def forward(self, fine_or_fusion_img: torch.tensor) -> torch.tensor:
        return self.net(fine_or_fusion_img)
