# -*- coding: utf-8 -*-
"""网络结构（docs/04-network-spec.md §1~3）。

Encoder: PIMoG U_Net_Encoder_Diffusion，消息长度 30→192、取值 {0,1}→{±1}；
Decoder: PIMoG Decoder/Extractor，输出长度 30→192；
Discriminator: PIMoG 原样。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from noise_layer import Identity, ScreenShooting

MSG_LEN = 192  # 水印序列长度 L


class ConvBNRelu(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1):
        super(ConvBNRelu, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layers(x)


class SingleConv(nn.Module):
    def __init__(self, inchannel, outchannel, s):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=s, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, s):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=s, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel))
        self.shortcut = nn.Sequential()
        if s != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=s, bias=False),
                nn.BatchNorm2d(outchannel))

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, num_channels=64):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            ConvBNRelu(3, num_channels),
            ConvBNRelu(num_channels, num_channels),
            ConvBNRelu(num_channels, num_channels),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.linear = nn.Linear(num_channels, 1)

    def forward(self, x):
        D = self.discriminator(x)
        D.squeeze_(3).squeeze_(2)
        D = self.linear(D)
        return D


class DoubleConv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.up(x)


class U_Net_Encoder_Diffusion(nn.Module):
    """PIMoG U-Net 编码器：消息序列经 Linear(192→256) 在 4 个上采样层级注入。"""

    def __init__(self, inchannel=3, outchannel=3, msg_len=MSG_LEN):
        super(U_Net_Encoder_Diffusion, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Globalpool = nn.MaxPool2d(kernel_size=4, stride=4)

        self.Conv1 = DoubleConv(inchannel, 16)
        self.Conv2 = DoubleConv(16, 32)
        self.Conv3 = DoubleConv(32, 64)

        self.Up4 = up_conv(64 * 3, 64)
        self.Conv7 = DoubleConv(64 * 3, 64)

        self.Up3 = up_conv(64, 32)
        self.Conv8 = DoubleConv(32 * 2 + 64, 32)

        self.Up2 = up_conv(32, 16)
        self.Conv9 = DoubleConv(16 * 2 + 64, 16)

        self.Conv_1x1 = nn.Conv2d(16, outchannel, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(msg_len, 256)
        self.Conv_message = DoubleConv(1, 64)

    def _expand(self, watermark, size):
        e = self.linear(watermark)
        e = e.view(-1, 1, 16, 16)
        e = F.interpolate(e, size=size, mode='bilinear')
        return self.Conv_message(e)

    def forward(self, x, watermark):
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)

        x6 = self.Globalpool(x4)
        x7 = x6.repeat(1, 1, 4, 4)
        # 瓶颈注入尺寸跟随 x4（128 输入 → 16×16；256 输入 → 32×32）
        em = self._expand(watermark, (x4.shape[2], x4.shape[3]))
        x4 = torch.cat((x4, x7, em), dim=1)

        d4 = self.Up4(x4)
        em = self._expand(watermark, (d4.shape[2], d4.shape[3]))
        d4 = torch.cat((x3, d4, em), dim=1)
        d4 = self.Conv7(d4)

        d3 = self.Up3(d4)
        em = self._expand(watermark, (d3.shape[2], d3.shape[3]))
        d3 = torch.cat((x2, d3, em), dim=1)
        d3 = self.Conv8(d3)

        d2 = self.Up2(d3)
        em = self._expand(watermark, (d2.shape[2], d2.shape[3]))
        d2 = torch.cat((x1, d2, em), dim=1)
        d2 = self.Conv9(d2)

        out = self.Conv_1x1(d2)
        return out


class Extractor(nn.Module):
    """PIMoG 残差提取头：自适应池化 → 1×16×16=256 → Linear(256→192)。

    任意输入分辨率可用：layer5 输出经 AdaptiveAvgPool2d 固定为 16×16
    （128 输入时 16×16 不变；256 输入时 32×32 → 16×16）。
    """

    def __init__(self, inchannel=64, msg_len=MSG_LEN):
        super(Extractor, self).__init__()
        self.layer1 = SingleConv(inchannel, 64, 1)
        self.layer2 = nn.Sequential(ResidualBlock(64, 64, 1), ResidualBlock(64, 64, 2))
        self.layer3 = nn.Sequential(ResidualBlock(64, 64, 1), ResidualBlock(64, 64, 2))
        self.layer4 = nn.Sequential(ResidualBlock(64, 64, 1), ResidualBlock(64, 64, 2))
        self.layer5 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((16, 16))
        self.linear = nn.Linear(256, msg_len)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.pool(out)
        out.squeeze_(1)
        out = out.view(-1, 1, 256)
        out = self.linear(out)
        out.squeeze_(1)
        return out


class Decoder(nn.Module):
    def __init__(self, msg_len=MSG_LEN):
        super(Decoder, self).__init__()
        self.extractor = Extractor(msg_len=msg_len)
        self.layer1 = nn.Sequential(
            SingleConv(3, 64, 1),
            SingleConv(64, 64, 1),
            SingleConv(64, 64, 1),
            ResidualBlock(64, 64, 1),
            ResidualBlock(64, 64, 1),
            ResidualBlock(64, 64, 1))

    def forward(self, x):
        x1 = self.layer1(x)
        message = self.extractor(x1)
        return message


class Encoder_Decoder(nn.Module):
    """编码器 + 噪声层 + 解码器的训练前向组合（PIMoG 同款接线）。

    distortion: 'Identity' | 'ScreenShooting'；crop 层由外部在 Noiser 之后插入。
    """

    def __init__(self, distortion='ScreenShooting', msg_len=MSG_LEN):
        super(Encoder_Decoder, self).__init__()
        self.Encoder = U_Net_Encoder_Diffusion(msg_len=msg_len)
        self.Decoder = Decoder(msg_len=msg_len)
        self.distortion = distortion
        if distortion == 'Identity':
            self.Noiser = Identity()
        elif distortion == 'ScreenShooting':
            self.Noiser = ScreenShooting()
        else:
            raise ValueError(f'未知 distortion: {distortion}')

    def forward(self, x, m):
        Encoded_image = self.Encoder(x, m)
        Noised_image = self.Noiser(Encoded_image)
        Decoded_message = self.Decoder(Noised_image.float())
        return Encoded_image, Noised_image, Decoded_message
