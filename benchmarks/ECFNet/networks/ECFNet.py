import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride,
        bias=True,
        norm=False,
        relu=True,
        transpose=False,
    ):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=bias,
                )
            )
        else:
            layers.append(
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=bias,
                )
            )
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())  # nn.ReLU(inplace=True)
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(
            *[
                nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
                # nn.LeakyReLU(negative_slope=0.1)
                # nn.ReLU(inplace=True)
                nn.GELU(),
            ]
        )

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDBlock(nn.Module):
    def __init__(self, in_channel, out_channel, nConvLayers=3):
        super(RDBlock, self).__init__()
        G0 = in_channel
        G = in_channel
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, out_channel, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()
        layers = [RDBlock(out_channel, out_channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()
        layers = [RDBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AFF1(nn.Module):
    def __init__(self, in_channel, out_channel, ffn_expansion_factor=2, bias=False):
        super(AFF1, self).__init__()
        hidden_features = int(in_channel * ffn_expansion_factor)
        self.project_in = nn.Conv2d(
            in_channel, hidden_features * 2, kernel_size=1, bias=bias
        )
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(
            hidden_features * 2, out_channel, kernel_size=1, bias=bias
        )

    def forward(self, x1, x2, x4, x8):
        x = torch.cat([x1, x2, x4, x8], dim=1)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x1 = F.sigmoid(x1) * x2
        x2 = F.sigmoid(x2) * x1

        x = torch.cat([x1, x2], dim=1)
        x = self.project_out(x)

        return x  # self.conv(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False),
        )

    def forward(self, x1, x2, x4, x8):
        x = torch.cat([x1, x2, x4, x8], dim=1)
        return self.conv(x)


class SCM(nn.Module):
    def __init__(self, out_plane, in_nc=3):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_nc, out_plane // 4, kernel_size=3, stride=1, relu=True),
            BasicConv(
                out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True
            ),
            BasicConv(
                out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True
            ),
            BasicConv(
                out_plane // 2, out_plane - in_nc, kernel_size=1, stride=1, relu=True
            ),
        )
        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, in_nc=3):
        super(SAM, self).__init__()
        self.conv1 = BasicConv(
            n_feat, n_feat, kernel_size=kernel_size, stride=1, relu=True
        )
        self.conv2 = BasicConv(
            n_feat, in_nc, kernel_size=kernel_size, stride=1, relu=False
        )
        self.conv3 = BasicConv(
            in_nc, n_feat, kernel_size=kernel_size, stride=1, relu=False
        )

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


class ECFNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, base_channel=None, num_res=8):
        super(ECFNet, self).__init__()

        base_channel = in_nc * 8 if base_channel is None else base_channel

        self.Encoder = nn.ModuleList(
            [
                EBlock(base_channel, num_res),
                EBlock(base_channel * 2, num_res),
                EBlock(base_channel * 4, num_res),
                EBlock(base_channel * 4, num_res),
            ]
        )

        self.feat_extract = nn.ModuleList(
            [
                BasicConv(in_nc, base_channel, kernel_size=3, relu=True, stride=1),
                BasicConv(
                    base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2
                ),
                BasicConv(
                    base_channel * 2,
                    base_channel * 4,
                    kernel_size=3,
                    relu=True,
                    stride=2,
                ),
                BasicConv(
                    base_channel * 4,
                    base_channel * 4,
                    kernel_size=3,
                    relu=True,
                    stride=2,
                ),
                BasicConv(
                    base_channel * 4,
                    base_channel * 4,
                    kernel_size=4,
                    relu=True,
                    stride=2,
                    transpose=True,
                ),
                BasicConv(
                    base_channel * 4,
                    base_channel * 2,
                    kernel_size=4,
                    relu=True,
                    stride=2,
                    transpose=True,
                ),
                BasicConv(
                    base_channel * 2,
                    base_channel,
                    kernel_size=4,
                    relu=True,
                    stride=2,
                    transpose=True,
                ),
                BasicConv(base_channel, out_nc, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.Decoder = nn.ModuleList(
            [
                DBlock(base_channel * 4, num_res),
                DBlock(base_channel * 4, num_res),
                DBlock(base_channel * 2, num_res),
                DBlock(base_channel, num_res),
            ]
        )

        self.Convs = nn.ModuleList(
            [
                BasicConv(
                    base_channel * 4 * 2,
                    base_channel * 4,
                    kernel_size=1,
                    relu=True,
                    stride=1,
                ),
                BasicConv(
                    base_channel * 2 * 2,
                    base_channel * 2,
                    kernel_size=1,
                    relu=True,
                    stride=1,
                ),
                BasicConv(
                    base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1
                ),
            ]
        )

        self.ConvsOut = nn.ModuleList(
            [
                SAM(base_channel * 4, in_nc=in_nc),
                SAM(base_channel * 4, in_nc=in_nc),
                SAM(base_channel * 2, in_nc=in_nc),
            ]
        )

        self.AFFs = nn.ModuleList(
            [
                AFF(base_channel * 11, base_channel * 1),
                AFF1(base_channel * 11, base_channel * 2),
                AFF1(base_channel * 11, base_channel * 4),
                AFF1(base_channel * 11, base_channel * 4),
            ]
        )
        self.FAM0 = FAM(base_channel * 4)
        self.FAM1 = FAM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM0 = SCM(base_channel * 4, in_nc=in_nc)
        self.SCM1 = SCM(base_channel * 4, in_nc=in_nc)
        self.SCM2 = SCM(base_channel * 2, in_nc=in_nc)

    def forward(self, x):
        # For development, we check if tensor size is multiple of 32
        _, _, W, H = x.shape

        x_2 = F.interpolate(x, scale_factor=0.5)  # 1, 4, 128, 128
        x_4 = F.interpolate(x_2, scale_factor=0.5)  # 1, 4, 64, 64
        x_8 = F.interpolate(x_4, scale_factor=0.5)  # 1, 4, 32, 32

        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)
        z8 = self.SCM0(x_8)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        res3 = self.Encoder[2](z)

        z = self.feat_extract[3](res3)
        z = self.FAM0(z, z8)
        z = self.Encoder[3](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z14 = F.interpolate(res1, scale_factor=0.25)
        z18 = F.interpolate(res1, scale_factor=0.125)

        z21 = F.interpolate(res2, scale_factor=2)
        z24 = F.interpolate(res2, scale_factor=0.5)
        z28 = F.interpolate(res2, scale_factor=0.25)

        z48 = F.interpolate(res3, scale_factor=0.5)
        z42 = F.interpolate(res3, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        z84 = F.interpolate(z, scale_factor=2)
        z82 = F.interpolate(z84, scale_factor=2)
        z81 = F.interpolate(z82, scale_factor=2)

        # print(z18.shape, z28.shape, z48.shape, z.shape)
        z = self.AFFs[3](z18, z28, z48, z)
        res3 = self.AFFs[2](z14, z24, res3, z84)
        res2 = self.AFFs[1](z12, res2, z42, z82)
        res1 = self.AFFs[0](res1, z21, z41, z81)

        z = self.Decoder[0](z)
        z, z_ = self.ConvsOut[0](z, x_8)
        z = self.feat_extract[4](z)
        outputs.append(z_)

        z = torch.cat([z, res3], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z, z_ = self.ConvsOut[1](z, x_4)
        z = self.feat_extract[5](z)
        outputs.append(z_)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z, z_ = self.ConvsOut[2](z, x_2)
        z = self.feat_extract[6](z)
        outputs.append(z_)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[2](z)
        z = self.Decoder[3](z)
        z = self.feat_extract[7](z)
        outputs.append(z + x)

        return outputs


if __name__ == "__main__":
    patch_size = 256
    channel = 4

    model = ECFNet(in_nc=channel, out_nc=channel).to("cuda:0")
    print(torch.cuda.memory_summary())
    time.sleep(5)

    print("-" * 50)
    print("#generator parameters:", sum(param.numel() for param in model.parameters()))
