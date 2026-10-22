# Phase 3 protocol: DRAFT, not locked, nothing run

Status: proposal for review. It becomes `PHASE3_PREREGISTRATION.md` only after the decisions marked **[decide]** are
settled. No Phase 3 test-set evaluation has been made.

## Question

Does the Phase 2 dropout model (B), trained only on *missing* optical, also hold up better than the full-input model (A)
when optical imagery is *cloud-corrupted*? This phase evaluates the existing models and does no new training.

## Fixed from Phase 2 (no changes)

- Checkpoints: `checkpoints/robust/p2_full_seed{1..10}.pt` and `p2_dropout_seed{1..10}.pt`, paired by seed.
- Split: 10 train / 4 validation / 10 official test cities. SAR inputs, labels and preprocessing are unchanged.
- Metric: pooled pixel counts per condition, giving F1, precision, recall, IoU, TP/FP/FN/TN and false-alarm rate.

## Corruption ladder (applied to S2 only; which dates is **[decide]**: post only, as primary, or both dates)

| level | target cloud fraction of the image | purpose |
|---|---|---|
| clean | 0 | reference (equals Phase 2 `full`) |
| light | ~10% | mild |
| moderate | ~30% | intermediate |
| heavy | ~60% | severe |
| missing | 100%, zero-filled | link to Phase 2 `no_o2` |

Fractions are fixed in advance. The masks are generated once with a fixed seed per city and level, saved to disk, and are
identical for both arms and all model seeds.

## How corrupted pixels are represented **[decide]**

1. **Real cloud masks (preferred):** binary or probability masks taken from a public annotated Sentinel-2 cloud dataset
   (e.g. CloudSEN12) and pasted onto OSCD tiles. Inside the mask, the S2 reflectance is replaced by a per-band cloud
   spectrum that is opaque or semi-transparent with alpha from the mask. Needs a download, and its licence must be recorded.
2. **Real cloudy imagery:** cloudy/clear S2 pairs (e.g. SEN12MS-CR). This is more realistic, but those scenes are not OSCD
   tiles, so it does not fit OSCD directly.
3. **Procedural only:** fractal-noise masks with a band-dependent cloud and haze model. This option can only be reported
   as a *simulated* robustness test.

Proposal: option 1 for masks and option 3's spectral model for pixel values, plus a thin-haze variant
(low alpha everywhere) as a separate condition. All of it is described as **simulated corruption with real cloud shapes**.
No result here is a claim about real cloudy acquisitions.

## Thresholds

The validation cities get the same corruption, from their own saved masks. For each arm, seed and level, the threshold is
argmax validation F1. Test cities are evaluated once per checkpoint and level. F1 at 0.5 is also reported. If both are cheap
to report, the clean-input validation threshold is reported too: a deployed model would not know the cloud level.
**[decide]** which one is primary. Proposal: the per-level validation threshold is primary, and the clean threshold is secondary.

## Analysis

- Curves: F1, precision, recall, IoU and false-alarm rate against severity, as mean ± sample SD over 10 seeds per arm.
- Primary: paired B − A F1 at each corrupted level, using a paired t-test plus Wilcoxon, with Holm correction across the
  corrupted levels. Claim rule as in Phase 2: Holm p < 0.05 and Δ > 0.
- Failure-mode analysis: false positives inside versus outside the cloud mask, to separate "clouds read as change" from
  "lost detections under clouds".
- Decision for Step 4: a new architecture (e.g. a reliability gate) is considered only if B shows a specific, repeatable
  failure at some level. One example would be false positives concentrated inside cloud masks that dropout does not remove.
