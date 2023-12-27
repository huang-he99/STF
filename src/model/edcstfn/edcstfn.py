import torch
import torch.nn as nn
import torch.nn.functional as F
import enum
from .ssim import msssim

NUM_BANDS = 6


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.ReplicationPad2d(1), nn.Conv2d(in_channels, out_channels, 3, stride=stride)
    )


def interpolate(inputs, size=None, scale_factor=None):
    return F.interpolate(
        inputs,
        size=size,
        scale_factor=scale_factor,
        mode='bilinear',
        align_corners=True,
    )


class FEncoder(nn.Sequential):
    def __init__(self):
        channels = [NUM_BANDS, 32, 64, 128]
        super(FEncoder, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(True),
            conv3x3(channels[2], channels[3]),
            nn.ReLU(True),
        )


class REncoder(nn.Sequential):
    def __init__(self):
        channels = [NUM_BANDS * 3, 32, 64, 128]
        super(REncoder, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(True),
            conv3x3(channels[2], channels[3]),
        )


class Decoder(nn.Sequential):
    def __init__(self):
        channels = [128, 64, 32, NUM_BANDS]
        super(Decoder, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(True),
            nn.Conv2d(channels[2], channels[3], 1),
        )


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.encoder = FEncoder()
        self.residual = REncoder()
        self.decoder = Decoder()

    def forward(self, inputs):
        # inputs[0] = interpolate(inputs[0], scale_factor=16)
        # inputs[-1] = interpolate(inputs[-1], scale_factor=16)
        prev_diff = self.residual(torch.cat((inputs[0], inputs[1], inputs[-1]), 1))

        if len(inputs) == 5:
            # inputs[2] = interpolate(inputs[2], scale_factor=16)
            next_diff = self.residual(torch.cat((inputs[2], inputs[3], inputs[-1]), 1))
            if self.training:
                prev_fusion = self.encoder(inputs[1]) + prev_diff
                next_fusion = self.encoder(inputs[3]) + next_diff
                return self.decoder(prev_fusion), self.decoder(next_fusion)
            else:
                one = inputs[0].new_tensor(1.0)
                epsilon = inputs[0].new_tensor(1e-8)
                prev_dist = torch.abs(prev_diff) + epsilon
                next_dist = torch.abs(next_diff) + epsilon
                prev_mask = one.div(prev_dist).div(
                    one.div(prev_dist) + one.div(next_dist)
                )
                prev_mask = prev_mask.clamp_(0.0, 1.0)
                next_mask = one - prev_mask
                result = prev_mask * (
                    self.encoder(inputs[1]) + prev_diff
                ) + next_mask * (self.encoder(inputs[3]) + next_diff)
                result = self.decoder(result)
                return result
        else:
            return self.decoder(self.encoder(inputs[1]) + prev_diff)


class CompoundLoss(nn.Module):
    def __init__(self, model, alpha=0.8, normalize=True):
        super(CompoundLoss, self).__init__()
        # self.pretrained = pretrained
        self.alpha = alpha
        self.normalize = normalize

        self.model = model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.encoder = nn.Sequential(
            self.model.conv1, self.model.conv2, self.model.conv3, self.model.conv4
        )

    def forward(self, prediction, target):
        _prediction, _target = self.encoder(prediction), self.encoder(target)
        return (
            F.mse_loss(prediction, target)
            + F.mse_loss(_prediction, _target)
            + self.alpha * (1.0 - msssim(prediction, target, normalize=self.normalize))
        )


class Sampling(enum.Enum):
    UpSampling = enum.auto()
    DownSampling = enum.auto()
    Identity = enum.auto()


class Upsample(nn.Module):
    def __init__(self, scale_factor=2):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, inputs):
        return F.interpolate(inputs, scale_factor=self.scale_factor)


class Conv3X3NoPadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv3X3NoPadding, self).__init__(
            in_channels, out_channels, 3, stride=stride, padding=1
        )


class Conv3X3WithPadding(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv3X3WithPadding, self).__init__(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3, stride=stride),
        )


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, sampling=None):
        layers = []

        if sampling == Sampling.DownSampling:
            layers.append(Conv3X3WithPadding(in_channels, out_channels, 2))
        else:
            if sampling == Sampling.UpSampling:
                layers.append(Upsample(2))
            layers.append(Conv3X3WithPadding(in_channels, out_channels))

        layers.append(nn.LeakyReLU(inplace=True))
        super(ConvBlock, self).__init__(*layers)


class AutoEncoder(nn.Module):
    def __init__(self, in_channels=NUM_BANDS, out_channels=NUM_BANDS):
        super(AutoEncoder, self).__init__()
        channels = (16, 32, 64, 128)
        self.conv1 = ConvBlock(in_channels, channels[0])
        self.conv2 = ConvBlock(channels[0], channels[1], Sampling.DownSampling)
        self.conv3 = ConvBlock(channels[1], channels[2], Sampling.DownSampling)
        self.conv4 = ConvBlock(channels[2], channels[3], Sampling.DownSampling)
        self.conv5 = ConvBlock(channels[3], channels[2], Sampling.UpSampling)
        self.conv6 = ConvBlock(channels[2] * 2, channels[1], Sampling.UpSampling)
        self.conv7 = ConvBlock(channels[1] * 2, channels[0], Sampling.UpSampling)
        self.conv8 = nn.Conv2d(channels[0] * 2, out_channels, 1)

    def forward(self, inputs):
        l1 = self.conv1(inputs)
        l2 = self.conv2(l1)
        l3 = self.conv3(l2)
        l4 = self.conv4(l3)
        l5 = self.conv5(l4)
        l6 = self.conv6(torch.cat((l3, l5), 1))
        l7 = self.conv7(torch.cat((l2, l6), 1))
        out = self.conv8(torch.cat((l1, l7), 1))
        return out
