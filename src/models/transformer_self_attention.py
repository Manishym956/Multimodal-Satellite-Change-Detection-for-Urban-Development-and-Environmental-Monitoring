"""Transformer Self-Attention model for asynchronous multimodal change detection.

Architecture — Model A: TransformerSelfAttention
=================================================
Input (asynchronous):
    SAR stream  : S1_T1 VV + S1_T2 VV  = 2 channels
    Optical stream: historical S2_T1    = 13 channels

Pipeline:
    1. Patch embedding (Conv2d, stride-4, non-overlapping 4×4 patches):
           SAR     : (B, 2,  H, W) → (B, 128, H/4, W/4)
           Optical : (B, 13, H, W) → (B, 128, H/4, W/4)
    2. Flatten spatial → token sequence:
           (B, 128, Hs, Ws) → (B, Hs*Ws, 128)   [T tokens per modality]
    3. Independent self-attention per modality:
           2 × TransformerBlock  (pre-LN, 4 heads, MLP ratio 4)
    4. Fuse: concatenate along channel dim after reshaping back to spatial map:
           (B, 128, Hs, Ws) cat (B, 128, Hs, Ws) → (B, 256, Hs, Ws)
    5. Lightweight convolutional decoder (4× upsampling to match input H,W):
           (B, 256, Hs, Ws) → (B, 1, H, W)  change logit

Key properties:
    - Handles arbitrary H,W: pads to multiple of patch_stride before embedding,
      crops back after decoding.
    - Each modality's self-attention is fully independent (no shared weights).
    - Decoder uses ConvTranspose2d and DoubleConv to produce dense output.
    - Output is a single-channel raw logit (apply sigmoid for probability).

IMPORTANT DISCLAIMER
--------------------
This is a research baseline architecture that combines standard Vision Transformer
self-attention with a simple convolutional decoder. It is NOT a reproduction of
any specific published paper and should not be labelled as such.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Utility: pad spatial dims to multiple of a stride
# ---------------------------------------------------------------------------

def _pad_to_multiple(
    x: torch.Tensor, multiple: int
) -> Tuple[torch.Tensor, int, int]:
    """Zero-pad x to make H and W divisible by *multiple*.

    Args:
        x:        (B, C, H, W)
        multiple: the required divisor

    Returns:
        (padded_x, orig_H, orig_W)
    """
    orig_h, orig_w = x.shape[2], x.shape[3]
    pad_h = (multiple - orig_h % multiple) % multiple
    pad_w = (multiple - orig_w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        # F.pad format: (left, right, top, bottom)
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, orig_h, orig_w


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    """Feed-forward MLP block used inside TransformerBlock.

    Shape: (B, T, D) → (B, T, D)
    """

    def __init__(self, dim: int, mlp_ratio: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = dim * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: LN → MHSA → residual → LN → MLP → residual.

    Notation:
        B = batch, T = tokens, D = embed_dim

    Args:
        dim:       token embedding dimension D
        num_heads: number of attention heads
        mlp_ratio: hidden-dim multiplier for the MLP
        dropout:   attention and MLP dropout probability
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # expects (B, T, D) not (T, B, D)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        # -- Self-attention with pre-norm --
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)  # Q=K=V=normed
        x = x + attn_out                                  # residual

        # -- MLP with pre-norm --
        x = x + self.mlp(self.norm2(x))                  # residual
        return x                                          # (B, T, D)


class CrossAttentionBlock(nn.Module):
    """Pre-norm Cross-Attention block.

    SAR tokens are query; optical tokens are key/value.
    After cross-attention, an independent MLP refines the SAR representation.

    Notation:
        B = batch, T = tokens, D = embed_dim

    Args:
        dim:       token embedding dimension D
        num_heads: number of attention heads
        mlp_ratio: hidden-dim multiplier for the MLP
        dropout:   attention and MLP dropout probability
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.norm_q = nn.LayerNorm(dim)   # normalise query (SAR tokens)
        self.norm_kv = nn.LayerNorm(dim)  # normalise key/value (optical tokens)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(
        self, x_sar: torch.Tensor, x_opt: torch.Tensor
    ) -> torch.Tensor:
        """Cross-attention: SAR queries attend to optical keys/values.

        Args:
            x_sar: (B, T, D) SAR token sequence (query)
            x_opt: (B, T, D) optical token sequence (key + value)

        Returns:
            updated SAR tokens (B, T, D)
        """
        q = self.norm_q(x_sar)
        kv = self.norm_kv(x_opt)
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)   # Q from SAR, K/V from optical
        x_sar = x_sar + attn_out                     # residual update to SAR stream
        x_sar = x_sar + self.mlp(self.norm_mlp(x_sar))  # MLP refinement
        return x_sar                                  # (B, T, D)


# ---------------------------------------------------------------------------
# Per-modality encoder: patch embed + N self-attention blocks
# ---------------------------------------------------------------------------

class ModalityEncoder(nn.Module):
    """Patch embedding + stacked self-attention blocks for one modality.

    Converts a spatial feature map to a sequence of patch tokens, applies
    *num_blocks* TransformerBlock layers, and reshapes back to spatial.

    Shapes:
        Input  : (B, in_channels, H, W)
        Output : (B, embed_dim, Hs, Ws)   where Hs = H // patch_stride
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 128,
        patch_stride: int = 4,
        num_blocks: int = 2,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_stride = patch_stride
        self.embed_dim = embed_dim

        # Non-overlapping patch projection via Conv2d
        # kernel_size == stride → each 4×4 spatial region becomes one token
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_stride, stride=patch_stride,
        )

        # Stack of independent self-attention blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads,
                             mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_blocks)
        ])

        self.norm = nn.LayerNorm(embed_dim)  # final layer-norm on token sequence

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        tokens_2d = self.patch_embed(x)           # (B, D, Hs, Ws)
        b, d, hs, ws = tokens_2d.shape

        # Flatten spatial → token sequence
        tokens = tokens_2d.flatten(2).transpose(1, 2)  # (B, Hs*Ws, D)

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)                 # (B, Hs*Ws, D)

        # Reshape back to 2-D spatial map
        tokens_2d = tokens.transpose(1, 2).reshape(b, d, hs, ws)  # (B, D, Hs, Ws)
        return tokens_2d


# ---------------------------------------------------------------------------
# Lightweight convolutional decoder
# ---------------------------------------------------------------------------

class _DoubleConv(nn.Module):
    """Conv3×3-BN-ReLU × 2 block (same as dual-stream U-Net building block)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TransformerDecoder(nn.Module):
    """Two-stage 2× upsampling decoder to recover full spatial resolution.

    Lifts from the patch grid (H/4, W/4) back to (H, W) using two successive
    ConvTranspose2d×2 steps, each followed by a DoubleConv.

    Input : (B, fused_dim, Hs, Ws)  where Hs = H // 4, Ws = W // 4
    Output: (B, out_ch,   H,  W)
    """

    def __init__(self, fused_dim: int = 256, mid_ch: int = 128, out_ch: int = 64) -> None:
        super().__init__()
        # Stage 1: (H/4, W/4) → (H/2, W/2)
        self.up1 = nn.ConvTranspose2d(fused_dim, mid_ch, kernel_size=2, stride=2)
        self.conv1 = _DoubleConv(mid_ch, mid_ch)

        # Stage 2: (H/2, W/2) → (H, W)
        self.up2 = nn.ConvTranspose2d(mid_ch, out_ch, kernel_size=2, stride=2)
        self.conv2 = _DoubleConv(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, fused_dim, Hs, Ws)
        x = self.conv1(self.up1(x))   # (B, mid_ch, Hs*2, Ws*2)
        x = self.conv2(self.up2(x))   # (B, out_ch, Hs*4, Ws*4) == (B, out_ch, H, W)
        return x


# ---------------------------------------------------------------------------
# Model A: TransformerSelfAttention
# ---------------------------------------------------------------------------

class TransformerSelfAttention(nn.Module):
    """Transformer Self-Attention model for asynchronous multimodal change detection.

    SAR and optical streams are independently encoded with patch embedding
    and per-modality self-attention, then fused by channel concatenation.
    The fused feature map is decoded back to full resolution.

    Asynchronous setup:
        SAR     (ch 0-1)   : S1_T1 VV + S1_T2 VV   (2 channels, both dates)
        Optical (ch 2-14)  : historical S2_T1 only   (13 channels, T1 only)
        S2_T2 is NEVER loaded or passed to this model.

    Args:
        sar_channels:   number of SAR input channels (default 2)
        opt_channels:   number of optical input channels (default 13)
        embed_dim:      patch embedding / token dimension (default 128)
        patch_stride:   spatial stride of the patch embedding Conv2d (default 4)
        num_blocks:     TransformerBlocks per modality encoder (default 2)
        num_heads:      attention heads per block (default 4)
        mlp_ratio:      MLP hidden-dim multiplier (default 4)
        dropout:        dropout rate in attention and MLP (default 0.0)
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
        num_blocks: int = 2,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.sar_channels = sar_channels
        self.opt_channels = opt_channels
        self.patch_stride = patch_stride

        # Independent per-modality encoders
        self.sar_encoder = ModalityEncoder(
            in_channels=sar_channels,
            embed_dim=embed_dim,
            patch_stride=patch_stride,
            num_blocks=num_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.opt_encoder = ModalityEncoder(
            in_channels=opt_channels,
            embed_dim=embed_dim,
            patch_stride=patch_stride,
            num_blocks=num_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        fused_dim = embed_dim * 2  # 256 after concatenating SAR + optical

        # Lightweight convolutional decoder: patch-grid → full resolution
        self.decoder = TransformerDecoder(
            fused_dim=fused_dim,
            mid_ch=embed_dim,       # 128
            out_ch=embed_dim // 2,  # 64
        )

        # Output head: 1-channel change logit
        self.head = nn.Conv2d(embed_dim // 2, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 15, H, W) concatenated input where:
               x[:, 0:2]  = SAR (S1_T1, S1_T2)
               x[:, 2:15] = Optical (S2_T1 13 bands)

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

        # Pad spatial dims to multiple of patch_stride so Conv2d divides evenly
        x_sar, orig_h, orig_w = _pad_to_multiple(x_sar, self.patch_stride)
        x_opt, _, _ = _pad_to_multiple(x_opt, self.patch_stride)

        # --- Per-modality self-attention encoding ---
        # Output: (B, embed_dim, H/4, W/4) for each modality
        sar_feats = self.sar_encoder(x_sar)  # (B, 128, Hs, Ws)
        opt_feats = self.opt_encoder(x_opt)  # (B, 128, Hs, Ws)

        # --- Channel concatenation fusion ---
        fused = torch.cat((sar_feats, opt_feats), dim=1)  # (B, 256, Hs, Ws)

        # --- Decode to full resolution ---
        out = self.decoder(fused)   # (B, 64, H, W)
        logits = self.head(out)     # (B,  1, H, W)

        # Crop back to original (unpadded) spatial dimensions
        return logits[:, :, :orig_h, :orig_w]


# ---------------------------------------------------------------------------
# Helper: count trainable parameters
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
