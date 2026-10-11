"""Dual Stream U-Net for binary change detection from SAR and optical imagery.

Architecture inspired by Hafner et al., "Sentinel-1 and Sentinel-2 Data Fusion
for Urban Change Detection Using a Dual Stream U-Net," IEEE GRSL 2021
(doi:10.1109/LGRS.2021.3119856).  Reference implementation: repos/2/DS_UNet.

IMPORTANT DISCLAIMER
--------------------
This is a *controlled re-implementation* that follows the structural design
described by Hafner et al. but is adapted to this project's conventions:

  * Project U-Net building blocks (DoubleConv, pad-to-multiple logic) are
    replaced by an independent implementation that mirrors Hafner's block
    design (Conv-BN-ReLU × 2, MaxPool down, ConvTranspose up with skip
    concatenation).
  * The two-argument forward signature used by Hafner's UNet.forward is not
    adopted because our training loop uses a single-tensor batch["image"].
    Instead, channel-splitting is done inside DualStreamUNet.forward so the
    training loop remains identical to the other baselines.
  * Hafner uses a variable-topology list (default [64, 128, 256, 512]).
    This implementation hard-codes an equivalent 4-level encoder/decoder pair
    with the same channel progression so parameter counts can be compared.
  * Hafner uses JaccardLikeLoss and augmentation (random flip/rotate).
    This baseline uses BCEWithLogitsLoss and the same augmentation-free
    protocol as the other project baselines so the comparison is fair.
  * This implementation has NOT been verified line-by-line against Hafner's
    original training run and should not be labelled an "exact reproduction."

Topology (matches Hafner base.yaml default [64, 128, 256, 512]):

    Per-modality encoder:
        inc:   in_ch  → 64           (no pooling, H × W)
        down1: 64     → 128          (H/2 × W/2)
        down2: 128    → 256          (H/4 × W/4)
        down3: 256    → 512          (H/8 × W/8)

    After concatenation (SAR + Optical) at each level:
        fused_s4: 1024  @ H/8 × W/8   (bottleneck)
        fused_s3:  512  @ H/4 × W/4
        fused_s2:  256  @ H/2 × W/2
        fused_s1:  128  @ H × W

    Shared decoder:
        up3: upsample(1024) + cat(skip=512) → DoubleConv(1024+512=1536, 512)
        up2: upsample(512)  + cat(skip=256) → DoubleConv(512+256=768,   256)
        up1: upsample(256)  + cat(skip=128) → DoubleConv(256+128=384,   128)

    Head:
        Conv2d(128, 1, 1)  → logit

Input tensor layout (single concatenated tensor fed by the training loop):
    channels 0      : S1 VV T1
    channels 1      : S1 VV T2
    channels 2–14   : S2 13 bands T1
    channels 15–27  : S2 13 bands T2
"""

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _DoubleConv(nn.Module):
    """Conv3x3-BN-ReLU × 2.  Matches Hafner's DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
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


class _Down(nn.Module):
    """MaxPool 2×2 then DoubleConv.  Matches Hafner's Down."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            _DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class _Up(nn.Module):
    """ConvTranspose2d upsample, concat skip, then DoubleConv.

    Args:
        up_in_channels:   channels in the feature map to be upsampled.
        skip_channels:    channels in the skip connection.
        out_channels:     channels after the DoubleConv.
    """

    def __init__(self, up_in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(up_in_channels, up_in_channels, kernel_size=2, stride=2)
        self.conv = _DoubleConv(up_in_channels + skip_channels, out_channels)

    def forward(self, x_up: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x_up = self.up(x_up)
        # Handle odd spatial sizes (same trick as Hafner's Up.forward)
        diff_y = x_skip.shape[2] - x_up.shape[2]
        diff_x = x_skip.shape[3] - x_up.shape[3]
        x_up = F.pad(x_up, (diff_x // 2, diff_x - diff_x // 2,
                             diff_y // 2, diff_y - diff_y // 2))
        return self.conv(torch.cat((x_skip, x_up), dim=1))


# ---------------------------------------------------------------------------
# Single-modality encoder (no decoder, no head)
# ---------------------------------------------------------------------------

class _ModalityEncoder(nn.Module):
    """U-Net encoder for one modality.  Returns 4 skip/feature maps."""

    def __init__(self, in_channels: int, base: int = 64) -> None:
        super().__init__()
        # Matches Hafner topology [64, 128, 256, 512]
        self.inc   = _DoubleConv(in_channels, base)      # (B, 64,  H,   W)
        self.down1 = _Down(base,     base * 2)           # (B, 128, H/2, W/2)
        self.down2 = _Down(base * 2, base * 4)           # (B, 256, H/4, W/4)
        self.down3 = _Down(base * 4, base * 8)           # (B, 512, H/8, W/8)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Returns (s1, s2, s3, s4) where s4 is the deepest feature map."""
        s1 = self.inc(x)     # 64  ch
        s2 = self.down1(s1)  # 128 ch
        s3 = self.down2(s2)  # 256 ch
        s4 = self.down3(s3)  # 512 ch
        return s1, s2, s3, s4


# ---------------------------------------------------------------------------
# Shared decoder
# ---------------------------------------------------------------------------

class _SharedDecoder(nn.Module):
    """Decoder operating on fused (SAR + Optical) skip connections.

    Receives the concatenated feature maps from both modality encoders:
        fused_s4: (B, 1024, H/8, W/8)  ← bottleneck cat
        fused_s3: (B,  512, H/4, W/4)
        fused_s2: (B,  256, H/2, W/2)
        fused_s1: (B,  128, H,   W)

    Channel accounting (b=64):
        up3: upsample(1024) + skip(512) → conv → 512    [in=1536, out=512]
        up2: upsample(512)  + skip(256) → conv → 256    [in=768,  out=256]
        up1: upsample(256)  + skip(128) → conv → 128    [in=384,  out=128]

    Output: (B, 128, H, W)
    """

    def __init__(self, base: int = 64) -> None:
        super().__init__()
        b = base
        # _Up(up_in, skip, out)
        self.up3 = _Up(b * 8 * 2, b * 4 * 2, b * 4 * 2)  # 1024, 512 → 512
        self.up2 = _Up(b * 4 * 2, b * 2 * 2, b * 2 * 2)  # 512,  256 → 256
        self.up1 = _Up(b * 2 * 2, b * 2,     b * 2)       # 256,  128 → 128

    def forward(
        self,
        fused_s4: torch.Tensor,   # (B, 1024, H/8, W/8)
        fused_s3: torch.Tensor,   # (B,  512, H/4, W/4)
        fused_s2: torch.Tensor,   # (B,  256, H/2, W/2)
        fused_s1: torch.Tensor,   # (B,  128, H,   W)
    ) -> torch.Tensor:
        d = self.up3(fused_s4, fused_s3)  # (B, 512, H/4, W/4)
        d = self.up2(d, fused_s2)         # (B, 256, H/2, W/2)
        d = self.up1(d, fused_s1)         # (B, 128, H,   W)
        return d


# ---------------------------------------------------------------------------
# Dual Stream U-Net
# ---------------------------------------------------------------------------

class DualStreamUNet(nn.Module):
    """Dual-stream U-Net baseline following Hafner et al. 2021.

    Accepts configurable SAR and optical channel counts to support both
    Scenario 1 (2 SAR + 26 optical = 28 ch) and Scenario 2 (4 SAR + 20 optical = 24 ch).
    Defaults are the Scenario 1 values; pass sar_channels / opt_channels to override.

    The network accepts a single pre-concatenated tensor:
        [0 : sar_channels]                        SAR channels
        [sar_channels : sar_channels+opt_channels] Optical channels

    Inputs whose H or W are not divisible by 8 are padded and cropped back.
    """

    # Scenario 1 defaults (2 SAR + 26 optical = 28 ch)
    SAR_CHANNELS = 2
    OPT_CHANNELS = 26

    def __init__(self, base_channels: int = 64,
                 sar_channels: int | None = None,
                 opt_channels: int | None = None) -> None:
        super().__init__()
        self.base_channels = base_channels
        self.pad_multiple  = 8

        self._sar_ch = sar_channels if sar_channels is not None else self.SAR_CHANNELS
        self._opt_ch = opt_channels if opt_channels is not None else self.OPT_CHANNELS

        b = base_channels
        self.sar_encoder = _ModalityEncoder(in_channels=self._sar_ch, base=b)
        self.opt_encoder = _ModalityEncoder(in_channels=self._opt_ch, base=b)
        self.decoder = _SharedDecoder(base=b)
        self.head = nn.Conv2d(b * 2, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, sar_channels+opt_channels, H, W) pre-concatenated tensor.
        Returns:
            logits: (B, 1, H, W) raw change map, before sigmoid.
        """
        # Pad to multiple of 8
        x, orig_h, orig_w = _pad_to_multiple(x, self.pad_multiple)

        # Split into SAR and Optical streams using instance channel counts
        x_sar = x[:, :self._sar_ch]
        x_opt = x[:, self._sar_ch: self._sar_ch + self._opt_ch]

        # Independent encoders
        sar_s1, sar_s2, sar_s3, sar_s4 = self.sar_encoder(x_sar)
        opt_s1, opt_s2, opt_s3, opt_s4 = self.opt_encoder(x_opt)

        # Fuse each level by concatenation
        fused_s4 = torch.cat((sar_s4, opt_s4), dim=1)   # (B, 1024, H/8, W/8)
        fused_s3 = torch.cat((sar_s3, opt_s3), dim=1)   # (B,  512, H/4, W/4)
        fused_s2 = torch.cat((sar_s2, opt_s2), dim=1)   # (B,  256, H/2, W/2)
        fused_s1 = torch.cat((sar_s1, opt_s1), dim=1)   # (B,  128, H,   W)

        # Shared decoder
        decoded = self.decoder(fused_s4, fused_s3, fused_s2, fused_s1)  # (B, 128, H, W)

        # Prediction head
        logits = self.head(decoded)  # (B, 1, H, W)

        # Crop back to original spatial dimensions
        return logits[:, :, :orig_h, :orig_w]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _pad_to_multiple(
    x: torch.Tensor, multiple: int
) -> tuple[torch.Tensor, int, int]:
    """Pad H and W up to the next multiple, return (padded, orig_H, orig_W)."""
    orig_h, orig_w = x.shape[-2], x.shape[-1]
    pad_h = (multiple - orig_h % multiple) % multiple
    pad_w = (multiple - orig_w % multiple) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, orig_h, orig_w


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
