"""Masked loss module for multimodal satellite change detection.
Ensures invalid sensor observations (e.g. SAR NaNs) do not contribute
to the loss numerator or denominator.
"""
import torch
from torch import nn
import torch.nn.functional as F


class MaskedBCEWithLogitsLoss(nn.Module):
    """BCEWithLogitsLoss with pixel-level validity masking.

    Formula:
        loss = sum(valid_mask * per_pixel_loss) / sum(valid_mask)

    Properties:
    1. Invalid pixels (mask == 0) contribute 0 to the numerator.
    2. Invalid pixels (mask == 0) do not contribute to the denominator.
    3. If sum(valid_mask) == 0 (batch has no valid pixels), returns 0.0 * logits.sum()
       to safely preserve autograd connectivity without altering parameter gradients.
    """

    def __init__(self, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits:     (B, 1, H, W) raw unnormalized logits
            targets:    (B, 1, H, W) binary change targets in {0.0, 1.0}
            valid_mask: (B, 1, H, W) validity mask in {0.0, 1.0} or bool
        Returns:
            Scalar loss tensor
        """
        per_pixel_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        mask = valid_mask.float()
        valid_count = mask.sum()
        if valid_count > 0:
            return (per_pixel_loss * mask).sum() / valid_count
        else:
            return logits.sum() * 0.0
