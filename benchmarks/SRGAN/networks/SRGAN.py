import math
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, io_channels=3):
        # upsample_block_num = int(math.log(scale_factor, 2))  # 0
        upsample_block_num = 0
        self.io_channels = io_channels

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(self.io_channels, 64, kernel_size=9, padding=4), nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, self.io_channels, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self, io_channels=3):
        self._is_full_backward_hook
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            CLBLock(io_channels, 64, 3, 1, 1),
            CBLBlock(64, 64, 3, 2, 1),
            CBLBlock(64, 128, 3, 1, 1),
            CBLBlock(128, 128, 3, 2, 1),
            CBLBlock(128, 256, 3, 1, 1),
            CBLBlock(256, 256, 3, 2, 1),
            CBLBlock(256, 512, 3, 1, 1),
            CBLBlock(512, 512, 3, 2, 1),
            nn.AdaptiveAvgPool2d(1),
            CLBLock(512, 1024, 1, 1, 0),
            nn.Conv2d(1024, 1, kernel_size=1),
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


class CLBLock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(CLBLock, self).__init__()
        self.conv = nn.Conv2d(
            in_channel, out_channel, kernel_size, stride, padding, bias=False
        )
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.lrelu(self.conv(x))


class CBLBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(CBLBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channel, out_channel, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channel)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.lrelu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
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
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
