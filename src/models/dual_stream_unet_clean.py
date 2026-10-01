"""Clean Asynchronous Dual Stream U-Net for multimodal change detection.

Architecture:
    SAR Encoder:
        Inputs: S1_T1 + S1_T2 = 2 channels
        Topology: 2 -> [64, 128, 256, 512]
    Optical Encoder:
        Inputs: historical S2_T1 only = 13 channels
        Topology: 13 -> [64, 128, 256, 512]
        NOTE: Directly accepts 13 channels with NO zero-filled placeholder channels.
    Fusion:
        Concatenation at every level (bottleneck + skips):
        - level 4: 512 + 512 = 1024
        - level 3: 256 + 256 = 512
        - level 2: 128 + 128 = 256
        - level 1: 64 + 64 = 128
    Shared Decoder:
        Identical to the synchronous Dual Stream decoder.
    Head:
        Conv2d(128, 1, 1) -> 1 change logit.

Total Input Channels: 15 (2 SAR + 13 Optical).
Total Trainable Parameters: 27,276,609 (exact difference of 7,488 weights from 13 unused input channels).
"""

import torch
import torch.nn.functional as F
from torch import nn

from src.models.dual_stream_unet import (
    _DoubleConv,
    _Down,
    _ModalityEncoder,
    _SharedDecoder,
    _pad_to_multiple,
    count_parameters,
)


class DualStreamUNetClean(nn.Module):
    """Clean Asynchronous Dual Stream U-Net.

    SAR branch receives S1_T1 and S1_T2 (2 channels).
    Optical branch receives historical S2_T1 only (13 channels).
    No zero-filled channels are used.
    """

    SAR_CHANNELS = 2   # S1 VV T1, S1 VV T2
    OPT_CHANNELS = 13  # S2 T1 (13 bands)
    TOTAL_CHANNELS = 15  # 2 + 13

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        self.base_channels = base_channels
        self.pad_multiple = 8  # 3 downsampling levels (2^3 = 8)

        b = base_channels
        self.sar_encoder = _ModalityEncoder(in_channels=self.SAR_CHANNELS, base=b)
        self.opt_encoder = _ModalityEncoder(in_channels=self.OPT_CHANNELS, base=b)
        self.decoder = _SharedDecoder(base=b)
        self.head = nn.Conv2d(b * 2, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, x_opt: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: If x_opt is None, x is a single (B, 15, H, W) tensor where:
                 x[:, 0:2] is SAR (S1_T1, S1_T2)
                 x[:, 2:15] is Optical (S2_T1 13 bands)
               If x_opt is provided, x is x_sar of shape (B, 2, H, W).
            x_opt: Optional (B, 13, H, W) optical tensor.
        Returns:
            logits: (B, 1, H, W) raw binary change logit before sigmoid.
        """
        if x_opt is None:
            if x.shape[1] != self.TOTAL_CHANNELS:
                raise ValueError(
                    f"Expected {self.TOTAL_CHANNELS} channels (2 SAR + 13 Optical), got {x.shape[1]}"
                )
            x_sar = x[:, :self.SAR_CHANNELS]
            x_opt = x[:, self.SAR_CHANNELS:]
        else:
            x_sar = x

        # Pad to multiple of 8
        x_sar, orig_h, orig_w = _pad_to_multiple(x_sar, self.pad_multiple)
        x_opt, _, _ = _pad_to_multiple(x_opt, self.pad_multiple)

        # Independent encoders
        sar_s1, sar_s2, sar_s3, sar_s4 = self.sar_encoder(x_sar)
        opt_s1, opt_s2, opt_s3, opt_s4 = self.opt_encoder(x_opt)

        # Multi-scale fusion by concatenation (identical feature dimensions to synchronous baseline)
        fused_s4 = torch.cat((sar_s4, opt_s4), dim=1)  # 512 + 512 = 1024
        fused_s3 = torch.cat((sar_s3, opt_s3), dim=1)  # 256 + 256 = 512
        fused_s2 = torch.cat((sar_s2, opt_s2), dim=1)  # 128 + 128 = 256
        fused_s1 = torch.cat((sar_s1, opt_s1), dim=1)  # 64 + 64 = 128

        # Shared decoder
        decoded = self.decoder(fused_s4, fused_s3, fused_s2, fused_s1)

        # Head
        logits = self.head(decoded)

        # Crop back to original dimensions
        return logits[:, :, :orig_h, :orig_w]
