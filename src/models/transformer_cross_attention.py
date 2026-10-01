"""Transformer Self + Cross Attention model for asynchronous multimodal change detection.

Architecture — Model B: TransformerCrossAttention
==================================================
This model extends TransformerSelfAttention (Model A) by adding a cross-attention
stage after the per-modality self-attention encoders.

The design follows standard multi-modal attention practices where one modality
queries information from the other.  SAR tokens act as Query; historical optical
tokens act as Key and Value.  This lets the SAR stream selectively attend to
optical context for improved change detection.

NOTE: Cross-attention between different sensor modalities for change detection
is well-established in the literature (e.g., Chen et al. 2021 ChangeFormer,
Bandara & Patel 2022 BIT).  This implementation does NOT claim novelty.
It is a controlled research baseline for comparison against the self-attention-
only variant and the CNN baselines.

Pipeline:
    1. Patch embedding (identical to Model A):
           SAR     : (B, 2,  H, W) → (B, 128, H/4, W/4)
           Optical : (B, 13, H, W) → (B, 128, H/4, W/4)
    2. Per-modality self-attention (2 blocks each, identical to Model A)
    3. Cross-attention (SAR-queries optical):
           q  = SAR tokens    (B, T, 128)
           k,v = optical tokens (B, T, 128)
           → updated SAR tokens  (B, T, 128)
    4. Decode the updated SAR representation:
           Reshape to (B, 128, Hs, Ws) → convolutional decoder → (B, 1, H, W)

Key difference from Model A:
    Model A fuses by *concatenating* both streams → fused_dim = 256
    Model B fuses by *cross-attention* (optical informs SAR) → SAR dim stays 128
    Therefore Model B has fewer decoder parameters and a slightly lower total
    parameter count than Model A.

IMPORTANT DISCLAIMER
--------------------
This is a research baseline. The cross-attention mechanism is a standard tool
and is not presented as a novel contribution.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

# Reuse all shared building blocks from transformer_self_attention.py
from src.models.transformer_self_attention import (
    CrossAttentionBlock,
    ModalityEncoder,
    TransformerDecoder,
    _DoubleConv,
    _pad_to_multiple,
    count_parameters,
)


# ---------------------------------------------------------------------------
# Model B: TransformerCrossAttention
# ---------------------------------------------------------------------------

class TransformerCrossAttention(nn.Module):
    """Transformer Self + Cross Attention model for asynchronous change detection.

    After independent per-modality self-attention, SAR tokens attend to
    historical optical tokens via multi-head cross-attention.  Only the
    updated SAR token stream is decoded to produce the change map.

    Asynchronous setup:
        SAR     (ch 0-1)   : S1_T1 VV + S1_T2 VV   (2 channels)
        Optical (ch 2-14)  : historical S2_T1 only   (13 channels)
        S2_T2 is NEVER loaded or passed to this model.

    Cross-attention roles:
        Query  = SAR tokens        (SAR features drive the output representation)
        Key    = optical tokens    (historical optical provides context)
        Value  = optical tokens

    Args:
        sar_channels:    SAR input channels (default 2)
        opt_channels:    optical input channels (default 13)
        embed_dim:       token dimension (default 128)
        patch_stride:    patch embedding stride (default 4)
        num_self_blocks: self-attention blocks per modality (default 2)
        num_cross_blocks: cross-attention blocks (default 1)
        num_heads:       attention heads (default 4)
        mlp_ratio:       MLP hidden-dim multiplier (default 4)
        dropout:         dropout rate (default 0.0)
    """

    SAR_CHANNELS: int = 2
    OPT_CHANNELS: int = 13
    TOTAL_CHANNELS: int = 15

    def __init__(
        self,
        sar_channels: int = 2,
        opt_channels: int = 13,
        embed_dim: int = 128,
        patch_stride: int = 4,
        num_self_blocks: int = 2,
        num_cross_blocks: int = 1,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.sar_channels = sar_channels
        self.opt_channels = opt_channels
        self.patch_stride = patch_stride

        # --- Per-modality self-attention encoders (same as Model A) ---
        self.sar_encoder = ModalityEncoder(
            in_channels=sar_channels,
            embed_dim=embed_dim,
            patch_stride=patch_stride,
            num_blocks=num_self_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.opt_encoder = ModalityEncoder(
            in_channels=opt_channels,
            embed_dim=embed_dim,
            patch_stride=patch_stride,
            num_blocks=num_self_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # --- Cross-attention: SAR queries historical optical context ---
        # After cross-attention, only the SAR stream (dim=embed_dim) is decoded.
        # One CrossAttentionBlock is the default; additional blocks stack residually.
        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_cross_blocks)
        ])

        self.post_cross_norm = nn.LayerNorm(embed_dim)

        # --- Decoder: operates on the updated SAR stream only (dim=embed_dim=128) ---
        # Note: fused_dim = embed_dim (not 2*embed_dim as in Model A) because
        # cross-attention preserves the SAR token dimension rather than concatenating.
        self.decoder = TransformerDecoder(
            fused_dim=embed_dim,        # 128 (SAR tokens after cross-attn)
            mid_ch=embed_dim // 2,      # 64
            out_ch=embed_dim // 4,      # 32
        )

        self.head = nn.Conv2d(embed_dim // 4, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 15, H, W) concatenated input:
               x[:, 0:2]  = SAR (S1_T1, S1_T2)
               x[:, 2:15] = Optical (historical S2_T1, 13 bands)

        Returns:
            logits: (B, 1, H, W) raw change logit
        """
        if x.shape[1] != self.TOTAL_CHANNELS:
            raise ValueError(
                f"Expected {self.TOTAL_CHANNELS} input channels "
                f"({self.sar_channels} SAR + {self.opt_channels} optical), "
                f"got {x.shape[1]}"
            )

        x_sar = x[:, :self.sar_channels]   # (B, 2,  H, W)
        x_opt = x[:, self.sar_channels:]   # (B, 13, H, W)

        # Pad to multiple of patch_stride
        x_sar, orig_h, orig_w = _pad_to_multiple(x_sar, self.patch_stride)
        x_opt, _, _ = _pad_to_multiple(x_opt, self.patch_stride)

        # --- Step 1: Per-modality self-attention ---
        # Each encoder outputs (B, embed_dim, Hs, Ws) spatial token map
        sar_feats = self.sar_encoder(x_sar)  # (B, 128, Hs, Ws)
        opt_feats = self.opt_encoder(x_opt)  # (B, 128, Hs, Ws)

        b, d, hs, ws = sar_feats.shape

        # Flatten to token sequences for attention
        # (B, D, Hs, Ws) → (B, Hs*Ws, D)
        sar_tokens = sar_feats.flatten(2).transpose(1, 2)  # (B, T, 128)
        opt_tokens = opt_feats.flatten(2).transpose(1, 2)  # (B, T, 128)

        # --- Step 2: Cross-attention (SAR queries optical) ---
        # Each CrossAttentionBlock: sar_tokens ← attend(sar_tokens → opt_tokens)
        for cross_block in self.cross_blocks:
            sar_tokens = cross_block(sar_tokens, opt_tokens)  # (B, T, 128)

        sar_tokens = self.post_cross_norm(sar_tokens)  # final layer-norm

        # Reshape back to 2-D spatial map
        # (B, T, D) → (B, D, Hs, Ws)
        sar_updated = sar_tokens.transpose(1, 2).reshape(b, d, hs, ws)  # (B, 128, Hs, Ws)

        # --- Step 3: Decode updated SAR representation ---
        out = self.decoder(sar_updated)   # (B, 32, H, W)
        logits = self.head(out)           # (B,  1, H, W)

        # Crop back to original spatial dimensions
        return logits[:, :, :orig_h, :orig_w]
