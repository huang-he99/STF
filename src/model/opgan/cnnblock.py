import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

## Reference: https://github.com/leftthomas/SRGAN


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels * up_scale**2, kernel_size=3, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        return x


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True
    )


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True
    )


class ResBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride=1, downsample=None, res_scale=1
    ):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out


class Shallow_Feature_Extractor(nn.Module):
    def __init__(self, in_feats, num_res_blocks, n_feats):
        super(Shallow_Feature_Extractor, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv_head = conv3x3(in_feats, n_feats)

        self.RBs = nn.ModuleList()
        for _ in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats))

        self.conv_tail = conv3x3(n_feats, n_feats)

    def forward(self, x):
        x = F.relu(self.conv_head(x))
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        x = x + x1
        return x


class Feature_Extractor(nn.Module):
    def __init__(self, in_channels=3, n_feats=64, n_lv=3):
        super(Feature_Extractor, self).__init__()
        self.n_lv = n_lv
        channels_list = [in_channels] + [
            (2**lv_index) * n_feats for lv_index in range(self.n_lv)
        ]
        self.feats_extractor = nn.ModuleList()
        for lv_index in range(n_lv):
            self.feats_extractor.append(
                torch.nn.Sequential(
                    nn.Conv2d(
                        in_channels=channels_list[lv_index],
                        out_channels=channels_list[lv_index + 1],
                        kernel_size=3,
                        padding=1,
                    ),
                    # nn.BatchNorm2d(self.channels_list[lv_index+1]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        in_channels=channels_list[lv_index + 1],
                        out_channels=channels_list[lv_index + 1],
                        kernel_size=3,
                        padding=1,
                    ),
                    # nn.BatchNorm2d(self.channels_list[lv_index+1]),
                    nn.ReLU(inplace=True),
                )
            )
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        lv_in = x
        lv_out_list = []
        for lv_index in range(self.n_lv):
            lv_out = self.feats_extractor[lv_index](lv_in)
            lv_in = self.MaxPool(lv_out)
            lv_out_list.append(lv_out)
        return lv_out_list


class Feature_Fusion(nn.Module):
    r"""
    Args:
        is_feat_concat (bool): way of feature combination True/False. False for sum; True for concatenation
    """

    def __init__(self, out_channels, is_feat_concat=True):
        super(Feature_Fusion, self).__init__()

        self.out_channels = out_channels

        if is_feat_concat:
            self.fusion = conv3x3(2 * self.out_channels, self.out_channels)
        else:
            self.fusion = conv3x3(self.out_channels, self.out_channels)

    def forward(self, x):
        return self.fusion(x)
