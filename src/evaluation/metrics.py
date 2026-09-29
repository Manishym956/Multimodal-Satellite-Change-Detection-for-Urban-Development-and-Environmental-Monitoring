"""Pixel counts and rates for binary change masks.

Predictions are probabilities after sigmoid. A pixel is predicted change when
its probability is greater than or equal to 0.5.

False alarm rate is

    FAR = FP / (FP + TN)

This is the fraction of unchanged ground-truth pixels that the model marks as
change. It is the false positive rate on the no-change class. It is not
1 - precision, which would be FP / (TP + FP).

A rate whose denominator is zero is undefined and is returned as None.
"""

from typing import Mapping

import torch

THRESHOLD = 0.5
FAR_DEFINITION = (
    "FAR = FP / (FP + TN): the fraction of unchanged pixels predicted as change."
)


def counts_from_probabilities(
    probability: torch.Tensor,
    target: torch.Tensor,
    threshold: float = THRESHOLD,
) -> dict[str, int]:
    """Count TP, TN, FP, and FN. probability is already passed through sigmoid."""
    predicted = probability >= threshold
    changed = target >= 0.5
    predicted = predicted.to(dtype=torch.bool)
    changed = changed.to(dtype=torch.bool)
    true_positive = int((predicted & changed).sum().item())
    true_negative = int((~predicted & ~changed).sum().item())
    false_positive = int((predicted & ~changed).sum().item())
    false_negative = int((~predicted & changed).sum().item())
    return {
        "tp": true_positive,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
    }


def counts_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = THRESHOLD,
) -> dict[str, int]:
    return counts_from_probabilities(torch.sigmoid(logits), target, threshold)


def rates_from_counts(counts: Mapping[str, int]) -> dict[str, float | None]:
    true_positive = counts["tp"]
    true_negative = counts["tn"]
    false_positive = counts["fp"]
    false_negative = counts["fn"]
    return {
        "precision": _ratio(true_positive, true_positive + false_positive),
        "recall": _ratio(true_positive, true_positive + false_negative),
        "f1": _f1(true_positive, false_positive, false_negative),
        "iou": _ratio(true_positive, true_positive + false_positive + false_negative),
        "false_alarm_rate": _ratio(false_positive, false_positive + true_negative),
    }


def metrics_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = THRESHOLD,
) -> dict[str, int | float | None]:
    counts = counts_from_logits(logits, target, threshold)
    return {**counts, **rates_from_counts(counts)}


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float | None:
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)
