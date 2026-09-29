"""Single-stream U-Net for binary change masks.

This is a standard encoder-decoder with skip connections. It is not the
DS-UNet architecture. The caller concatenates the two Sentinel-1 dates into
one tensor of shape (batch, 2, height, width). The returned tensor is
logits of shape (batch, 1, height, width), before sigmoid.

Inputs whose height or width is not divisible by 16 are padded inside
forward and cropped back, so a full city image keeps its original size.
"""

import torch
import torch.nn.functional as F
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Binary segmentation U-Net. Default input has 2 channels and 1 logit channel."""

    def __init__(self, in_channels: int = 2, out_channels: int = 1, base_channels: int = 64):
        super().__init__()
        channels = base_channels
        self.pad_multiple = 16
        self.enc1 = DoubleConv(in_channels, channels)
        self.enc2 = DoubleConv(channels, channels * 2)
        self.enc3 = DoubleConv(channels * 2, channels * 4)
        self.enc4 = DoubleConv(channels * 4, channels * 8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(channels * 8, channels * 16)
        self.up4 = nn.ConvTranspose2d(channels * 16, channels * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(channels * 16, channels * 8)
        self.up3 = nn.ConvTranspose2d(channels * 8, channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(channels * 8, channels * 4)
        self.up2 = nn.ConvTranspose2d(channels * 4, channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(channels * 4, channels * 2)
        self.up1 = nn.ConvTranspose2d(channels * 2, channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(channels * 2, channels)
        self.head = nn.Conv2d(channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, height, width = _pad_to_multiple(x, self.pad_multiple)
        skip1 = self.enc1(x)
        skip2 = self.enc2(self.pool(skip1))
        skip3 = self.enc3(self.pool(skip2))
        skip4 = self.enc4(self.pool(skip3))
        hidden = self.bottleneck(self.pool(skip4))
        decoded = self.dec4(torch.cat((self.up4(hidden), skip4), dim=1))
        decoded = self.dec3(torch.cat((self.up3(decoded), skip3), dim=1))
        decoded = self.dec2(torch.cat((self.up2(decoded), skip2), dim=1))
        decoded = self.dec1(torch.cat((self.up1(decoded), skip1), dim=1))
        logits = self.head(decoded)
        return logits[:, :, :height, :width]


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
    height, width = x.shape[-2:]
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    if pad_height or pad_width:
        x = F.pad(x, (0, pad_width, 0, pad_height))
    return x, height, width


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
