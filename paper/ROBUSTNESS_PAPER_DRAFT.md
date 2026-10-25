# Clouds Are Not Missing Data: A Staged Robustness Study of SAR–Optical Urban Change Detection on OSCD with Pre-Specified Analyses

**Manish Y. M.**
*[Institution]*
manishym946@gmail.com

*Internal draft for mentor review, 2026-10-23; not for circulation. Every number in this draft is taken from the audited study reports in `experiments/robust/`
(`REPORT.md`, `PHASE3_REPORT.md`, `PHASE4A_REPORT.md`, `PHASE4ABIS_REPORT.md`, `PHASE4B_REPORT.md`, `PHASE5S_REPORT.md`,
`PHASE5_REPORT.md`) and checked against them by
`paper/check_robustness_paper_numbers.py` (every number occurs in a source) and `paper/check_robustness_paper_tables.py` (every
results-table cell matches its source value). Items marked [verify] need checking before submission.*

---

## Abstract

SAR–optical change detectors are usually trained and tested with all images present, yet in practice the post-change
optical image is often partly cloudy rather than absent. We study this gap with the dual-stream U-Net (DS-UNet) of Hafner et al. on the
OSCD urban change benchmark (Sentinel-1 VV and Sentinel-2), in seven steps. For each step, hypotheses, test conditions and the primary
analysis were written down before its test evaluation (an internal pre-registration; §3.5), and each uses 10 paired training seeds,
a frozen validation-only selection protocol and an independent audit of its reported statistics (§3.5).
(1) Whole-modality dropout training (B) shows no detectable full-input cost at a validation-selected threshold (Δ = −0.003, 95% CI
[−0.037, +0.031]; at a fixed 0.5 threshold it costs 0.086) and improves F1 with the post-change optical image zero-filled by +0.241
relative to full-input training (A).
(2) Under simulated opaque clouds with real cloud shapes, both models fail: 10% cloud cover reduces B's F1 from 0.553 to 0.329,
and at 30% cover B's F1 is 0.191, against 0.098 for a detector that labels every pixel as change. Cloud-valued pixels are
read as change (false-alarm rate under the cloud 0.96–1.00).
(3) Zero-filling the cloud pixels with an oracle mask restores B to 0.539 / 0.516 / 0.499 F1 at 10 / 30 / 60% cover,
recovering 87–94% of the loss; the full-input model A recovers only 16–40%.
(4) With imperfect masks, over-masking is almost free (−0.003 to −0.010 F1 at 30% cover for 2–8 px dilation), while
under-masking is expensive (−0.090 to −0.250 for missed clouds, −0.100 to −0.243 for eroded edges).
(5) Training with cloud-shaped spatial zero-fill (C) gives no detectable difference from whole-modality dropout on any of the 7
primary conditions (all |Δ| < 0.02 F1, Holm p ≥ 0.55), and does not reduce the under-masking failure.
(6) Dilating the imperfect masks by a validation-selected 16 px recovers most of the loss from missed cloud edges (B at 30% cover:
+0.085 to +0.194 F1) but not from missed whole clouds (|Δ| ≤ 0.002).
(7) With a real cloud detector (s2cloudless) on composites that paste real catalogue cloud pixels into the OSCD post-change
images, a recall-biased operating point (95% cloud recall on validation) gives B higher F1 than the most accurate mask at 30% and
60% cover (+0.023, +0.057) but stays below the oracle mask (−0.020 to −0.055) and retains 0.740 of the oracle gain at 30% cover,
just below the pre-specified 0.75. It reaches that recall by masking 60–67% of the clear test pixels, and a fallback mask that
removes almost the whole post-change image gives B similar F1 (0.478).
With synthetic mask errors, the main remaining error for this late-fusion architecture comes from cloud pixels that the mask lets
through; with the real detector, heavy over-masking also costs F1. Of the configurations we tested, zero-fill under a
recall-biased mask plus modality-dropout training is the most effective, but with this
detector much of its benefit comes from discarding most of the cloudy image rather than from locating the cloud. We did not
compare against SAR-to-optical feature reconstruction [19]. All results are on OSCD with one architecture and one
detector-and-zero-fill pipeline; the clouds are simulated or composited, never real cloudy acquisitions, and the 10 seeds are
training replicates on the same 10 test cities.

**Keywords:** change detection, SAR–optical fusion, cloud cover, missing modality, modality dropout, cloud masking, OSCD,
pre-registration, reproducibility

---

## 1. Introduction

Optical satellite images are often unusable at the date a change must be assessed, because clouds cover part or all of the
scene. SAR is unaffected by cloud, which motivates SAR–optical fusion for change detection [1]. Most fusion models, however,
are trained and evaluated with every image available. Two different failure cases are often treated as one:
the post-change optical image can be **missing** (no acquisition, or one discarded as unusable), or it can be **present but
partly cloudy**. A model that copes with the first case need not cope with the second.

Our earlier multi-seed study on OSCD measured the cost of a missing post-change optical image [17] and recommended modality
dropout as future work. This paper carries out that programme with one strong, published architecture, the DS-UNet of Hafner
et al. [1], and asks six questions in sequence, each internally pre-registered before its test evaluation (§3.5):

- **Q1 (Phase 2):** does whole-modality dropout training make the model robust to a missing post-change optical image, and what does it cost on full input?
- **Q2 (Phase 3):** does that robustness carry over to a post-change image that is partly covered by (simulated) opaque cloud?
- **Q3 (Steps 4A and 4A-bis):** does zero-filling the cloud pixels at inference help, with an oracle mask and with realistic mask errors?
- **Q4 (Step 4B):** does training with cloud-shaped spatial zero-fill improve on whole-modality dropout?
- **Q5 (Step 5S):** does dilating an under-masking mask (biasing it toward recall) recover the under-masking loss?
- **Q6 (Step 5):** with a real cloud detector on composite clouds made from real cloud pixels, does a recall-biased operating point beat the most accurate one?

The answers are, in order: yes, at no detectable full-input cost; no; yes with a good mask, and with synthetic mask errors the mask
should err toward over-masking; no detectable improvement; for missed cloud edges but not for missed whole clouds; and yes at 30% and 60% cover, but
partly by discarding most of the image, and below the oracle mask. The practical conclusion is that, in this setting, a new fusion
architecture is not yet warranted: in our synthetic-mask experiments, the dominant remaining error is cloud-valued pixels reaching
the model, and with the real detector, over-masking also costs F1.

**Contributions.** This is a measurement study, not a new method. We contribute (i) a controlled, multi-seed separation of the
*missing* and *partly cloudy* cases for one published SAR–optical architecture; (ii) evidence that inference-time zero-fill under a cloud
mask works well only for a dropout-trained model; (iii) a quantification of the asymmetric cost of over- and under-masking, and of how
far recall-biased dilation repairs it; (iv) a null result for cloud-shaped spatial dropout training; and (v) a test of the
recall-biased masking recommendation with a real detector on composite clouds, which shows that it helps but partly by removing the
optical image. Each step's analysis was fixed in advance, and its statistics were recomputed by an independent audit script (§3.5).

Because OSCD is small (24 cities), we report every result with 10 paired seeds, 95% confidence intervals and Holm-corrected
tests, select every checkpoint and threshold on validation cities only, and treat the 10 seeds explicitly as training replicates
on one fixed test set, not as independent geographic samples. For every step except Step 4A, an independent audit script
re-derived the reported statistics from raw confusion counts; Step 4A's per-seed F1 values are re-checked by the Step 4A-bis audit.

## 2. Related Work

**SAR–optical change detection.** Hafner et al. [1] fuse Sentinel-1 and Sentinel-2 in a dual-stream U-Net with late fusion and
report urban change detection on OSCD [13]. Siamese fully convolutional networks [2] and U-Net variants [3] are common
optical baselines; reviews [6] cover the broader field.

**Missing modalities.** Modality dropout, randomly removing an input modality during training, was introduced for multimodal
gesture recognition [18] and is widely used to make fusion models usable when one input is absent. Closest to our setting, Hafner and
Ban [19] address multi-temporal urban mapping with an optical modality partly missing because of clouds. They reconstruct optical
features from SAR when optical input is absent, and report that this outperforms a zero-fill baseline and a SAR-only model. Our study
examines the zero-fill route in more detail (dropout training, cloud-valued vs zero-filled input, mask errors) and does not compare with
reconstruction; whether reconstruction also degrades under imperfect cloud masks is an open question.

**Clouds.** Real cloud masks for Sentinel-2 are available in the Sentinel-2 Cloud Mask Catalogue [20], which we use for cloud
shapes. Operational cloud detectors such as s2cloudless [22] are imperfect; an intercomparison of ten cloud-masking algorithms on four
reference datasets [23] documents substantial errors. Steps 4A-bis and 5S model detector errors synthetically; Step 5 uses
s2cloudless [22] on composite clouds.

**Evaluation practice.** Seed variance can reverse model rankings in small benchmarks [16, 17]. We follow a pre-registration
discipline: each step's hypotheses, conditions, test family and decision rule were written down and hashed before the step's
primary test evaluation. In Phase 2, the test F1 of an earlier step-budget sweep had been viewed; that choice was discarded and
redone on validation (Appendix A, item 1). These are internal pre-registrations, not public registrations (§3.5).

## 3. Methods

### 3.1 Data and split

OSCD [13] with Sentinel-2 L1C (13 bands, /10000, clipped to [0, 1]) and the Sentinel-1 VV rasters released with DS-UNet [1]
(cubic resize to the label grid). Each sample has, per date, VV plus 13 optical bands. We use a holdout protocol: 10 of the
14 official training cities for training, the other 4 (aguasclaras, beihai, bercy, paris) for validation, and the 10 official
test cities for testing. The test cities never enter training or validation, which is asserted in code and re-checked in every run file.
The change prevalence on the test cities is 0.052 (the precision of a detector that labels every pixel as change), and that
detector's F1 is 0.098.

### 3.2 Model and training recipe

The network and loss are the official DS-UNet files (`networks.DualStreamUNet`, topology [64, 128, 256, 512], late fusion by
1×1 convolution; `jaccard_like_loss`), loaded at run time from commit 8bfa4be. Training: Adam, learning rate 5e-5, weight decay
5e-4, batch 64, 32×32 importance crops (20 candidates weighted by changed pixels + 5), random flips and 90° rotations,
2500 steps, validation every 25 steps. The step budget was chosen on validation score only (390 steps: 0.354; 1000: 0.382;
2500: 0.405; mean of 3 seeds; Appendix A, item 1). Training is bitwise deterministic on our hardware (PyTorch 2.5.1+cu124, cuDNN deterministic,
one RTX 4060 laptop GPU).

As a reference point, the official 390-step protocol on all 14 training cities gives test F1 0.557 ± 0.035 (10 seeds,
training-city threshold), below the published 0.600 [1]. We could not establish whether 0.600 is a single run or a mean, so
this comparison is unresolved; our conclusions concern differences between arms under one shared protocol.

### 3.3 Training arms

| arm | name | per-sample training conditions | seeds |
|---|---|---|---|
| A | full-only | full input | 1–10 |
| B | whole-modality dropout | full 0.50; post-change S2 zero (`no_o2`) 0.25; both S2 dates zero (`no_opt`) 0.25 | 1–10 |
| C | spatial dropout | full 0.50; post-change S2 zero under a cloud-shaped mask (`spatial`) 0.25; `no_o2` 0.125; `no_opt` 0.125 | 1–10 |

The mixtures were fixed a priori and not tuned. C keeps B's full-input share and replaces half of B's whole-image dropout with
spatial dropout, so the expected fraction of zero-filled post-change pixels falls from 0.50 (B) to 0.370 (C). The C − B
comparison therefore mixes the effect of spatial masks with the effect of less whole-image dropout. C's training masks
are 20,000 32×32 windows from 153 Cloud Mask Catalogue scenes (mean cloud fraction 0.48), excluding all 42 scenes used for any
Phase 3 validation or test mask. (The Step 5 composites, built later, do reuse bank scenes; §4.7.)

**Selection (validation cities only).** Checkpoint: the step with the highest mean validation F1 over the conditions the arm
trains on (A: full; B and C: full, `no_o2`, `no_opt`). Threshold: the validation-F1 argmax over {0, 0.01, …, 1} for the
selected checkpoint. From Phase 3 on, the **primary threshold is each checkpoint's frozen clean-validation threshold**, applied
unchanged to every corrupted condition, since a deployed model does not know the cloud level.

### 3.4 Test conditions

All corruptions act on the post-change Sentinel-2 image (t2) only; SAR and the pre-change optical image are unchanged.

- **Missing** (`no_o2`): t2 optical set to zero. **Copy-forward** (`o2_copy`): t2 optical replaced by the t1 optical image. **No optical** (`no_opt`): both optical dates set to zero.
- **Cloud** (c10 / c30 / c60): a real CLOUD mask from the Sentinel-2 Cloud Mask Catalogue [20], one per city and level, adjusted
  to cover exactly 10 / 30 / 60% of the city's pixels; inside it, all 13 bands are replaced by a fixed opaque cloud spectrum
  (per-band median of 400,000 thick-cloud pixels from 8 catalogue scenes). Clouds are uniform, hard-edged and shadowless.
- **Oracle zero-fill** (z10 / z30 / z60): the cloud pixels are set to zero instead, using the true mask M.
- **Imperfect masks** (4A-bis): the cloud image is zero-filled under a detector mask P derived from M: eroded by 2 / 4 / 8 px
  (missed cloud edges), dilated by 2 / 4 / 8 px (over-masked clear pixels), or with 10 / 25 / 50% of whole clouds dropped
  (missed clouds). At 30% cover, erosion misses 12 / 23 / 43% of the cloud pixels and dilation masks 5 / 11 / 21% of the clear pixels.
- **Dilated imperfect masks** (5S): each eroded and missed-cloud mask is dilated by a disk of radius r ∈ {0, 1, 2, 3, 4, 6, 8,
  12, 16} px, with the Phase 3 clouds unchanged. One radius r* for all masks, levels and arms is chosen on validation cities only
  (B's mean validation F1 over the 18 under-masking conditions): r* = 16 px, the largest radius in the grid.
- **Composite clouds and a real detector** (Step 5): real cloud pixels replace the flat spectrum. For each validation and test
  city, level and replicate (3 per level; 126 composites, 36 validation and 90 test), a 20 m window of a Cloud Mask Catalogue [20]
  scene whose CLOUD fraction is within ±10% (relative) of the level is upsampled to 10 m; inside its CLOUD mask M, the
  post-change image is replaced by the scene's 13 top-of-atmosphere bands (clipped to [0, 1]). Cloud shadows are not transplanted.
  The 22 scenes the catalogue rates least reliable are excluded and the remaining 491 are split into validation and test pools by
  hash, so validation and test composites never share a scene. The detector is s2cloudless 1.7.3 [22] on 10 bands of the
  post-change image. A detector mask P is (cloud probability ≥ t) dilated by a disk of radius r, over t ∈ {0.05, 0.10, …, 0.90}
  and the radius grid above. Operating points were chosen on the validation composites only, before any change model saw a test
  composite: **rb95** (primary), the most precise point with pooled cloud recall ≥ 0.95 (t 0.25, r 8: validation recall 0.954,
  masking 0.458 of clear pixels); **bal**, the point with the highest mask F1 (t 0.65, r 6: mask F1 0.836); rb90 and rb99 by the
  same rule (rb99 reached its target at no grid point, so the rule's fallback, the highest-recall point t 0.05, r 16, was used and
  flagged); and the library's default mask. Each mask is applied with zero-fill. The detector is also run on the clean
  post-change images, to measure what false detections cost when there is no cloud.

Zero is the fill value used in these experiments. It is not an intrinsic missing-data token: the models treat it as missing only
to the extent that training showed them zero-filled inputs.

### 3.5 Statistics and integrity

F1, precision, recall, IoU and false-alarm rate are computed from pooled pixel counts over the 10 test cities. Each step has one
primary family of paired comparisons over seeds 1–10 (two-sided paired t-test, Holm correction [21]), reported with the mean
difference, its 95% CI, d_z, the number of seeds favouring each arm, and a Holm-corrected Wilcoxon signed-rank test (smallest
attainable p with n = 10 is 0.0020). Secondary analyses form their own Holm families and are labelled descriptive. With 10 pairs,
the minimum detectable effect at 80% power is about 1.00 SD of the paired differences at α = 0.05, and at the first Holm step
1.41 SD for Step 4B (α = 0.05/7, roughly 0.04–0.07 F1), 1.37 SD for Step 5S (α = 0.05/6) and 1.46 SD for Step 5 (α = 0.05/9);
non-significant results are not evidence of equivalence. The confidence intervals and p-values describe variation across training
seeds on one fixed set of 10 test cities; they do not include the sampling variability of the test cities themselves.

**Pre-specification and its evidence.** For each step, a pre-registration file (hypotheses, conditions, primary family, decision guidance,
and sha256 hashes of the code and inputs) was written before that step's first test evaluation. These are internal files in the project
repository, not entries in a public registry. The evidence for their timing differs by step: the Step 4B pre-registration was committed
to version control before the first training run, and the Step 5S and Step 5 pre-registrations before their test evaluations (for Step 5,
the validation-selected operating points, the smoke test and the test detector run were appended and committed before any change
model was evaluated on a test composite); for Phases 2–4A-bis, the evidence is file modification times relative to the first run
files, and the code hashes recorded in every run file. The Step 4A and 4A-bis pre-registrations cite validation-only smoke tests for which
no log exists. Each evaluation enforces anchors, for example that re-evaluated clean counts equal the
training-time counts exactly. An independent audit script recomputes every reported statistic from raw counts (Phase 2: 579 checks;
Phase 3: 167; Step 4A-bis: 1338; Step 4B: 432; Step 5S: 826; Step 5: 731; all with 0 failures). Step 4A has no audit script of
its own; the Step 4A-bis audit re-checks its per-seed F1 values, but not its paired statistics or recovery ratios. Before Step 4B training, a parity run of the spatial-dropout
code with the spatial share set to zero reproduced a B checkpoint bit-for-bit. Deviations are listed in Appendix A.

## 4. Results

### 4.1 Whole-modality dropout handles a missing image at no detectable full-input cost (Phase 2)

Test F1 at the validation-selected threshold (mean ± SD over 10 seeds; primary family B − A, Holm over 4):

| condition | A full-only | B dropout | Δ (B − A) | 95% CI | B > A | Holm p |
|---|---|---|---|---|---|---|
| full | 0.556 ± 0.029 | 0.553 ± 0.030 | −0.003 | [−0.037, +0.031] | 5/10 | 0.8449 |
| post optical missing (`no_o2`) | 0.190 ± 0.068 | 0.431 ± 0.034 | +0.241 | [+0.176, +0.305] | 10/10 | 5.8e-05 |
| copy-forward (`o2_copy`) | 0.314 ± 0.034 | 0.438 ± 0.038 | +0.124 | [+0.087, +0.160] | 10/10 | 9.5e-05 |
| no optical (`no_opt`) | 0.397 ± 0.106 | 0.440 ± 0.027 | +0.043 | [−0.036, +0.123] | 5/10 | 0.4964 |

B is better with the post-change image missing and with copy-forward imputation, which neither arm saw in training. The `no_opt`
difference is not supported; it comes mostly from one collapsed A seed. On full input there is no detectable cost at the
validation-selected threshold, although differences below about 0.035 F1 cannot be excluded. At a fixed threshold of 0.5, B is
lower on full input (−0.086, secondary): B's probabilities shift upward, so the dropout model needs its threshold chosen on
validation data.

### 4.2 Partial cloud is a different failure, and dropout does not fix it (Phase 3)

Test F1 at the frozen clean threshold (B − A, Holm over 4):

| condition | A full-only | B dropout | Δ (B − A) | Holm p | B > A |
|---|---|---|---|---|---|
| clean | 0.556 ± 0.029 | 0.553 ± 0.030 | | | |
| c10 | 0.285 ± 0.015 | 0.329 ± 0.008 | +0.044 | 9.6e-05 | 10/10 |
| c30 | 0.165 ± 0.007 | 0.191 ± 0.012 | +0.026 | 0.0010 | 10/10 |
| c60 | 0.123 ± 0.002 | 0.134 ± 0.011 | +0.011 | 0.0144 | 10/10 |
| missing | 0.142 ± 0.050 | 0.478 ± 0.030 | +0.336 | 9.5e-08 | 10/10 |

B degrades significantly less than A at every cloud level, but the advantage is small next to the damage. Ten percent cloud cover costs
A about 0.27 F1 and B about 0.22 F1; at 30–60% both are close to the all-change F1 of 0.098. **For B, a partly cloudy post-change
image is far worse than none:** 0.329 F1 at 10% cover against 0.478 with the post-change image missing. (The `missing` value differs
from Phase 2's 0.431 only because Phase 3 uses the frozen clean threshold; the predictions are identical.)

The failure mechanism is visible directly. At the clean threshold, almost every pixel under a cloud is predicted as change: the
false-alarm rate inside the mask rises from about 0.03–0.05 to 0.96–1.00 for both arms. False alarms also spill into clear pixels
nearby (outside-mask false-alarm rate for A 0.025 → 0.058 / 0.134 / 0.199 at c10 / c30 / c60; for B 0.042 → 0.052 / 0.075 /
0.095). Dropout taught B to ignore zero-filled optical input, not cloud-valued optical input.

![Phase 3: F1 versus cloud cover](../experiments/robust/phase3/curves.png)

### 4.3 Zero-filling under the cloud recovers most of the loss, mainly for the dropout model (Step 4A)

Replacing the cloud values by zero under the true mask (oracle) improves F1 for both arms at every level (Holm p < 0.05 for all 6
tests, 10/10 seeds), but by very different amounts:

| arm | level | F1 cloud | F1 zero-fill | Δ | recovery of the cloud loss |
|---|---|---|---|---|---|
| A | 10% | 0.285 | 0.393 | +0.108 | 0.403 ± 0.189 |
| A | 30% | 0.165 | 0.261 | +0.097 | 0.249 ± 0.168 |
| A | 60% | 0.123 | 0.192 | +0.069 | 0.161 ± 0.132 |
| B | 10% | 0.329 | 0.539 | +0.210 | 0.939 ± 0.016 |
| B | 30% | 0.191 | 0.516 | +0.325 | 0.899 ± 0.033 |
| B | 60% | 0.134 | 0.499 | +0.365 | 0.872 ± 0.041 |

For B, the false-alarm rate inside the mask returns to its clean value (≈ 0.05) and the spill-over disappears; B at 60% cover with
the oracle mask (0.499) is above B with the whole image missing (0.478). A reads the zero-filled region as change too (false-alarm
rate inside the mask ≈ 0.51). The remedy works well only together with dropout training. B's remaining gap is lost evidence rather than false
alarms: its recall inside the mask falls from about 0.73 to 0.55–0.61 when it must rely on SAR and the pre-change image.

### 4.4 With synthetic mask errors, the mask should err toward over-masking (Step 4A-bis)

For B at 30% cover, the primary family compares each imperfect mask with the oracle mask (Holm over 9). Every error type costs F1
(Holm p < 0.05, worse than the oracle in 10/10 seeds), but the costs are very unequal:

| mask | missed cloud | over-masked clear | B F1 | Δ vs oracle | 95% CI | retained gain |
|---|---|---|---|---|---|---|
| oracle (z30) | 0 | 0 | 0.516 | | | |
| dilate2 | 0.000 | 0.051 | 0.513 | −0.003 | [−0.003, −0.002] | 0.99 |
| dilate4 | 0.000 | 0.107 | 0.511 | −0.005 | [−0.006, −0.005] | 0.98 |
| dilate8 | 0.000 | 0.214 | 0.506 | −0.010 | [−0.012, −0.009] | 0.97 |
| erode2 | 0.116 | 0.000 | 0.416 | −0.100 | [−0.141, −0.060] | 0.70 |
| erode4 | 0.232 | 0.000 | 0.329 | −0.187 | [−0.220, −0.153] | 0.43 |
| erode8 | 0.431 | 0.000 | 0.273 | −0.243 | [−0.272, −0.213] | 0.26 |
| miss10 | 0.100 | 0.000 | 0.426 | −0.090 | [−0.103, −0.077] | 0.73 |
| miss25 | 0.250 | 0.000 | 0.348 | −0.168 | [−0.188, −0.149] | 0.49 |
| miss50 | 0.500 | 0.000 | 0.266 | −0.250 | [−0.276, −0.225] | 0.23 |

Retained gain = (F1_mask − F1_cloud) / (F1_oracle − F1_cloud), with F1_cloud = 0.191.

Over-masking is almost free for B: masking 21% of the clear pixels costs 0.010 F1. Under-masking is expensive whatever its shape,
and missed edges and missed whole clouds cost similar amounts per missed pixel. On missed-cloud pixels, B's false-alarm rate is
0.909–0.962 for every missed-cloud and erode4/8 variant: unmasked cloud pixels are almost all called change. The same pattern
holds at 10% and 60% cover and under the other threshold rules. Over-masking does hurt the full-input model A: its false-alarm
rate on over-masked clear pixels rises from ≈ 0.024 to 0.385–0.497. An exploratory, not pre-registered partition by the true
mask shows that a thin rim of unmasked cloud is enough to bring back the Phase 3 spill-over: for B with erode4, the false-alarm
rate on clear pixels outside the cloud is 0.073, the same as with no fill at all (0.075), against 0.045 with the oracle mask.

![Step 4A-bis: F1 under imperfect masks](../experiments/robust/phase4abis/curves.png)

### 4.5 Spatial dropout training does not add robustness (Step 4B)

C − B test F1 at the frozen clean threshold (Holm over 7):

| condition | B | C | Δ (C − B) | 95% CI | C > B | Holm p |
|---|---|---|---|---|---|---|
| clean | 0.553 ± 0.030 | 0.564 ± 0.034 | +0.011 | [−0.024, +0.046] | 7/10 | 1.0 |
| z10 | 0.539 ± 0.028 | 0.553 ± 0.033 | +0.014 | [−0.019, +0.048] | 7/10 | 1.0 |
| z30 | 0.516 ± 0.028 | 0.529 ± 0.032 | +0.013 | [−0.020, +0.047] | 6/10 | 1.0 |
| z60 | 0.499 ± 0.030 | 0.508 ± 0.028 | +0.009 | [−0.024, +0.041] | 6/10 | 1.0 |
| c30, erode4 | 0.329 ± 0.029 | 0.312 ± 0.034 | −0.018 | [−0.050, +0.015] | 2/10 | 1.0 |
| c30, dilate4 | 0.511 ± 0.028 | 0.522 ± 0.030 | +0.011 | [−0.021, +0.043] | 6/10 | 1.0 |
| c30, miss25 | 0.348 ± 0.007 | 0.341 ± 0.006 | −0.007 | [−0.015, +0.001] | 3/10 | 0.55 |

All 7 primary comparisons fall under the pre-specified label "no detectable difference", and C shows no clean regression. None of
the 27 other mask conditions or the 4 Phase 2 conditions differs after Holm correction. Descriptively, C is lower than B on the
whole-modality conditions (`no_opt` −0.028, `o2_copy` −0.039 at the validation-selected threshold, Holm p 0.14), consistent with
its halved whole-image dropout, and C still calls almost every unmasked cloud pixel change (false-alarm rate under the cloud with
no fill 0.994, against 0.961 for B). C's precision and recall vary more across seeds than B's.

Exploratory and not tested (the conditions share models and pixels): C − B is negative in 19 of the 21 conditions in which cloud-valued
pixels reach the model and positive in 12 of the 13 conditions with complete zero-fill. If real, this would mean C is slightly
better when the mask is complete and slightly worse when it is not. These differences are around 0.01 F1, below what 10 seeds can resolve.

![Step 4B: C versus B](../experiments/robust/phase4b/curves.png)

### 4.6 Dilation repairs missed edges, not missed clouds (Step 5S)

B at 30% cover, test F1 with each under-masking mask P and with P dilated by r* = 16 px (primary family: paired two-sided t-test,
Holm over 6; Holm-corrected Wilcoxon p in `PHASE5S_REPORT.md`), and the
remaining difference from the oracle mask after dilation (secondary):

| mask | B F1, P | B F1, P dilated by r* | Δ | 95% CI | Δ > 0 | Holm p | Δ vs oracle at r* |
|---|---|---|---|---|---|---|---|
| erode2 | 0.416 ± 0.050 | 0.500 ± 0.027 | +0.085 | [+0.044, +0.125] | 10/10 | 0.0041 | −0.016 |
| erode4 | 0.329 ± 0.029 | 0.500 ± 0.027 | +0.171 | [+0.138, +0.204] | 10/10 | 4.3e-06 | −0.016 |
| erode8 | 0.273 ± 0.018 | 0.468 ± 0.020 | +0.194 | [+0.172, +0.216] | 10/10 | 6.0e-08 | −0.048 |
| miss10 | 0.426 ± 0.013 | 0.424 ± 0.015 | −0.002 | [−0.005, +0.001] | 4/10 | 0.2382 | −0.092 |
| miss25 | 0.348 ± 0.007 | 0.348 ± 0.008 | +0.000 | [−0.001, +0.002] | 5/10 | 0.7405 | −0.168 |
| miss50 | 0.266 ± 0.010 | 0.267 ± 0.011 | +0.002 | [+0.001, +0.003] | 9/10 | 0.0261 | −0.249 |

Dilation is labelled "dilation better" for all three eroded masks, with every seed improving: retained gain for erode4 rises from
0.43 to 0.95, and after dilation the eroded masks are within 0.048 F1 of the oracle. For missed whole clouds it makes no
practical difference: miss10 and miss25 show no detectable difference, and miss50 meets the pre-specified label "dilation better"
with an effect of 0.002 F1 that the Holm-corrected Wilcoxon test does not support (p 0.059). Dilation can only reach a missed cloud
that lies near a detected one. At 10% and 60% cover (secondary), the eroded masks gain +0.044 to +0.210 and the missed-cloud
masks +0.000 to +0.017. For the full-only model A, dilating the missed-cloud masks lowers F1 in 9 of 9 contrasts, consistent with A
reading zero-filled clear pixels as change (§4.4).

The validation score was still rising at r* = 16 px, the edge of the grid, so a larger radius might have scored higher. Over-dilation
was cheap on test: at r* every eroded mask is within 0.014 F1 of its own peak over the radius grid (descriptive; the test curve was not
used to choose r*). Clouds and mask errors in this step are both synthetic.

### 4.7 A real cloud detector on composite clouds: recall bias helps, partly by discarding the image (Step 5)

This step replaces the flat cloud spectrum by real catalogue cloud pixels pasted into the OSCD post-change images, and the
synthetic masks by s2cloudless masks (§3.4). These composites are not real cloudy acquisitions: thin cloud carries the foreign
surface from its source scene, there are no matching shadows, and the atmosphere outside the cloud is unchanged. Primary family:
B, test F1 at the frozen clean threshold, paired two-sided t-test, Holm over 9 (Holm-corrected Wilcoxon p in `PHASE5_REPORT.md`).

| contrast | Δ | 95% CI | Δ > 0 | Holm p | label |
|---|---|---|---|---|---|
| c10: rb95 − bal | −0.009 | [−0.019, +0.001] | 3/10 | 0.0672 | no detectable difference |
| c10: rb95 − oracle | −0.055 | [−0.065, −0.045] | 0/10 | 4.4e-06 | rb95 worse |
| c10: rb95 − cloud | +0.036 | [+0.017, +0.055] | 10/10 | 0.0056 | rb95 better |
| c30: rb95 − bal | +0.023 | [+0.008, +0.037] | 9/10 | 0.0119 | rb95 better |
| c30: rb95 − oracle | −0.053 | [−0.067, −0.039] | 0/10 | 6.7e-05 | rb95 worse |
| c30: rb95 − cloud | +0.150 | [+0.125, +0.175] | 10/10 | 2.1e-06 | rb95 better |
| c60: rb95 − bal | +0.057 | [+0.037, +0.078] | 10/10 | 5.6e-04 | rb95 better |
| c60: rb95 − oracle | −0.020 | [−0.024, −0.016] | 0/10 | 6.8e-06 | rb95 worse |
| c60: rb95 − cloud | +0.257 | [+0.223, +0.291] | 10/10 | 3.4e-07 | rb95 better |

**Detector masks on the test composites and B's mean test F1 (descriptive).** Cloud recall and the fraction of clear pixels masked
are computed against M with no change model; rb99 is the flagged fallback mask.

| level | rb95 cloud recall | rb95 clear masked | bal cloud recall | bal clear masked | B cloud | B bal | B rb95 | B rb99 | B oracle | rb95 retained gain |
|---|---|---|---|---|---|---|---|---|---|---|
| c10 | 0.892 | 0.596 | 0.532 | 0.069 | 0.447 | 0.492 | 0.483 | 0.478 | 0.538 | 0.35 ± 0.21 |
| c30 | 0.837 | 0.615 | 0.555 | 0.087 | 0.315 | 0.442 | 0.465 | 0.478 | 0.518 | 0.74 ± 0.06 |
| c60 | 0.963 | 0.673 | 0.747 | 0.094 | 0.217 | 0.417 | 0.474 | 0.478 | 0.494 | 0.93 ± 0.02 |

The recall-biased mask beats the most accurate one at 30% and 60% cover and is better than no fill at every level, but it stays
below the oracle mask at every level. Under the pre-specified decision guidance, the first branch (a working remedy) needs rb95
better than bal at 30% and a mean retained gain of at least 0.75; B's is 0.740. The third branch needs a retained gain below 0.5
or rb95 worse than bal. **No pre-specified branch applies exactly**, and we do not round 0.740 up to the threshold.

Three descriptive results qualify the gain. First, the detector misses more cloud on test than on validation: the operating point
chosen for 0.954 validation recall reaches 0.892 and 0.837 at 10% and 30% cover, so 11–16% of test cloud pixels reach the model,
which is the failure Steps 4A-bis and 5S found expensive. Second, rb95 reaches its recall by masking 0.596–0.673 of the clear test
pixels, far more than the 2–8 px dilations that were nearly free in Step 4A-bis. Its F1 outside the cloud falls from 0.543 (oracle)
to 0.488 at 30% cover, and applied to clean images it costs B 0.054 F1 (Holm p < 0.05). Third, rb99, which masks 0.992–0.996 of the
clear pixels and so is essentially B without the post-change optical image, gives 0.478 at every level, above rb95 at 30% cover.
The gain over bal is therefore consistent with B falling back on SAR and the pre-change image, not evidence that the detector
locates cloud well. The remaining gap to the oracle combines missed cloud pixels and over-masked clear pixels; this analysis
does not separate the two.

Secondary families: for A, rb95 is worse than bal at every level (−0.148, −0.083, −0.032; Holm p < 0.05) and worse than no fill at
10% and 30% cover. C follows B, and C − B shows no difference after Holm (Δ −0.000 to +0.015); C trained on cloud shapes from the
scenes behind 56 of the 90 test composites, so its results are flagged. Unfilled, the composite clouds cost B less than the flat
Phase 3 spectrum did (0.315 against 0.191 at 30% cover), but the two corruptions differ in more than realism and are not compared
statistically.

## 5. Discussion

**Missing and cloudy are different problems.** Phase 2 and Phase 3 together show that robustness to a missing image does not
transfer to a partly corrupted one. A dropout-trained model learns that a zero-filled optical image carries no information. It
does not learn that a bright, flat, cloud-valued patch is not a change. In this model, cloud-valued pixels act as strong evidence
of change, both under the cloud and, through the receptive field, in clear pixels nearby.

**In these experiments, the input-side remedy did most of the work.** Zero-filling cloud pixels turns the cloud problem back into the missing-data
problem that dropout training already solves. With a perfect mask this recovers 87–94% of the cloud loss. With an imperfect mask, the
outcome depends almost entirely on the direction of the error: a mask that covers 21% of the clear scene in addition to the cloud
costs B only 0.010 F1, while a mask that misses a quarter of the cloud costs 0.168. This asymmetry suggests tuning the cloud
detector for recall rather than for balanced accuracy, and only for a model trained with dropout; the full-input model reads
zero-filled clear pixels as change, so over-masking hurts it.

**Recall bias works, but its cost depends on the detector.** Dilation repairs missed cloud edges almost completely and cannot reach
missed whole clouds (Step 5S). With s2cloudless on composite clouds (Step 5), the recall-biased mask beats the most accurate one at
30% and 60% cover, but it buys its recall by masking most of the clear image, it still misses 11–16% of test cloud pixels at 10% and
30% cover, and a mask that removes almost the whole image does about as well. The small, nearly free over-masking of the synthetic
study does not describe this detector: here, over-masking means discarding most of the post-change optical image, and costs
0.054 F1 on clear images. For a dropout-trained model this trade is still better than leaving cloud pixels in the image, but the
gain should not be read as good cloud localisation. These conclusions are specific to this detector, its threshold-plus-dilation
operating points and zero-fill; a detector with better recall at low over-masking could behave differently.

**Zero-fill augmentation did not add robustness.** Spatial dropout teaches the model about zero-filled regions with cloud
shapes, but B already handles zero-filled regions well, and the residual failure is cloud values that the mask lets through. C never
sees cloud values in training, so the null result is what the Step 4B pre-registration expected. We expect that closing the remaining gap requires the
model to see or be told about cloud-valued pixels, for example training that includes cloud-valued corruption, or an explicit
per-pixel validity channel. Neither was tested here, and this study cannot rule out that a different architecture would be less
sensitive to unmasked cloud. Nor did we compare zero-fill with SAR-to-optical feature reconstruction [19], the main alternative
fill strategy; that comparison, under the same mask errors, is a natural next step.

**Architecture decision.** Under the pre-specified decision guidance, these results do not justify a new fusion architecture now.
Step 5 falls between its pre-specified branches: on these composites, the recall-biased real detector helps B, but retains 0.740 of the oracle gain at 30%
cover, not the 0.75 the first branch requires. The remaining candidates, each to be pre-registered separately, are (i) training with
cloud-valued corruption or a validity-mask input channel, which target the cloud pixels that any detector lets through, and (ii) a
test on real cloudy acquisitions, which is needed before any claim of robustness to real clouds. The validity channel is the minimal
architecture change. A reliability-gated fusion architecture would be justified only if these leave a specific, measured failure.

**Operating point.** Dropout-trained models shift their output probabilities upward and need their decision threshold chosen on
validation data; at a fixed 0.5 threshold, B loses 0.086 F1 on full input. Thresholds also vary widely between seeds (C's frozen
clean thresholds range from 0.50 to 0.97). Deployments should calibrate the threshold on held-out data from the target region.

## 6. Limitations

- **Simulated and composite corruption, never real cloudy acquisitions.** In Phases 3–5S the cloud shapes are real, but the clouds
  are opaque, uniform, hard-edged and shadowless, built from one fixed spectrum. In Step 5 the cloud pixels are real catalogue pixels,
  but pasted from other scenes: thin cloud carries the foreign surface, there are no matching shadows, the cloud edge is a hard
  boundary between two scenes, and the atmosphere outside the cloud is unchanged. In every step only the post-change date is
  corrupted. The haze condition (Phase 3, secondary) is the only test of partial transparency with the flat spectrum. None of
  these results measures performance on actual cloudy satellite acquisitions.
- **Synthetic mask errors, then one detector pipeline.** Erosion, dilation and whole-cloud removal (Steps 4A-bis, 5S) do not model a
  specific detector. Step 5 tests one detector (s2cloudless at 10 m on 10 bands), operating points of the form threshold plus disk
  dilation chosen on 36 validation composites, and zero-fill; other detectors, mask rules or fill strategies may behave differently.
  Detector false detections on clean OSCD images may include real, unlabelled cloud, and the catalogue's semi-automatic 20 m masks
  are upsampled, so their boundary errors count as detector errors.
- **Grid-boundary selections.** Step 5S's r* (16 px) and Step 5's rb99 fallback lie at the edge of their grids.
- **Resumed composites.** 89 of the 126 composite files were kept from interrupted build runs and verified by mask, shape, data type
  and value range, not byte for byte (Appendix A).
- **Zero as a fill value.** Zero is the fill value used here, applied to [0, 1]-scaled reflectance. It is not an intrinsic missing-data token.
- **Seeds are not geography.** The 10 seeds are training replicates evaluated on the same 10 OSCD test cities. They measure training
  variability; the p-values do not support claims about other regions, sensors or seasons.
- **Small benchmark, limited power.** OSCD has 24 cities; checkpoints and thresholds are selected on 4 validation cities. Effects below
  about 0.04–0.07 F1 can be missed in the Step 4B comparison.
- **Confounded Step 4B contrast.** C differs from B both by adding spatial dropout and by halving whole-image dropout. In Step 5, C
  trained on cloud shapes from the scenes behind 56 of the 90 test composites, so its Step 5 results are flagged.
- **Checkpoint selection differs between arms by design.** A selects its checkpoint on full-input validation F1, B and C on the mean over
  full, `no_o2` and `no_opt`. Part of the A–B difference may come from the selection rule rather than the training mixture.
- **Internal pre-registration.** The pre-registrations are repository files, not public registrations, and for Phases 2–4A-bis their
  timing rests on file times (§3.5).
- **Single benchmark and single fill strategy.** Only OSCD and only zero-fill were studied; reconstruction-based methods [19] may behave differently.
- **One architecture, one dropout mixture per arm.** The mixtures were fixed a priori and not tuned. The results may differ for other
  fusion designs (e.g. early fusion or attention-based models).
- **Baseline gap.** Our reproduction of the official DS-UNet protocol (0.557 ± 0.035) is below the published 0.600, and two small code
  differences from the official training loop remain (batch sampling with replacement; first-orbit SAR at evaluation).

## 7. Conclusion

In a seven-step, internally pre-registered study of SAR–optical urban change detection on OSCD with 10 paired seeds per arm, we
found that
(1) whole-modality dropout makes a DS-UNet robust to a missing post-change optical image at no detectable full-input cost;
(2) that robustness does not extend to partly cloudy images, because cloud-valued pixels are read as change;
(3) zero-filling the cloud pixels restores most of the lost F1 for the dropout-trained model, and much less for the full-input model;
(4) with synthetic mask errors, the cloud mask should err toward over-masking, because unmasked cloud pixels are expensive and small
over-masking is nearly free;
(5) training with cloud-shaped spatial zero-fill adds no detectable robustness;
(6) dilating an under-masking mask repairs missed cloud edges but not missed whole clouds; and
(7) with a real detector on composite clouds, a recall-biased mask beats the most accurate one at 30% and 60% cover but stays below
the oracle mask, and much of its gain comes from discarding most of the post-change optical image.
For practitioners, the most effective configuration we tested is a modality-dropout-trained model with a validation-calibrated threshold
and zero-fill under a recall-biased cloud mask, with the caveat that, with this detector, high recall means masking most of the
clear image. These findings are for one architecture on OSCD with simulated and composite clouds; whether they hold on real cloudy
acquisitions is the most important open question.

## Reproducibility statement

Code, pre-registrations, run files, logs, analysis tables, audits and reports are in the repository (`experiments/robust/`,
branch `robust-phase4`, commits `eead015`, `25fe13b`, `5dfaa17`; Step 5S `cde5434`, `3c5d85a`, `f56ba71`; Step 5 `d9e29ff`,
`5f5be9c`, `3ed2fb7`). Checkpoints, raw count arrays, composites and detector outputs are excluded from version control and
listed with sha256 hashes in `experiments/robust/ARTIFACT_SHA256_2026-10-22_post4b.txt` and
`experiments/robust/ARTIFACT_SHA256_2026-10-23_post5.txt`. The composites can be rebuilt from the public Cloud Mask Catalogue [20]
with `build_phase5_composites.py` and checked against the committed manifest. Each pre-registration's hash table can be
re-verified with `verify_prereg_hashes.py`. Training is bitwise deterministic on the hardware used.

## Appendix A: Deviations and gaps (as recorded in the step reports)

1. **Phase 2 step budget.** A 1000-step budget was first proposed after the test F1 of a 3-seed step sweep had been viewed. That choice
   was discarded and the budget re-chosen on validation score (2500 steps). Seeds 1–3 of A share the configuration of a sweep run whose
   test F1 had been seen; they were retrained from scratch, reproduced identically, and no Phase 2 decision used them.
2. **Phase 3 masks.** The first mask build used an absolute cloud-fraction tolerance (±0.05), which could nearly double a small cloud.
   The masks were rebuilt with a relative tolerance before any model saw a mask; the superseded masks are kept and unused. A missing
   `model.eval()` in the Phase 3 evaluation script was fixed after a validation-only smoke test, before any test evaluation.
3. **Steps 4A and 4A-bis.** Both pre-registrations cite a validation-only smoke test for which no log exists; those claims are unverified.
   In both steps, the evaluation anchors (exact reproduction of stored counts and thresholds) are the available evidence that the pipeline
   was correct. In Step 4A-bis, the evaluation printed per-seed test F1 while running; the analysis script's hash equals the registered hash. The analysis crashed on
   its final console print and was re-run unchanged. Two source files were hashed in every run file but missing from the pre-registration's
   hash table; their hashes are unchanged since Phase 3. Inside/outside-mask metrics and IoU were added as a labelled exploratory addendum.
4. **Phase 3 report wording.** `PHASE3_REPORT.md` describes the all-change F1 (≈ 0.098) as "the change prevalence". The prevalence
   is 0.052; 0.098 is the F1 of the all-change detector. The report is left unchanged; the correction is recorded in
   `experiments/robust/PHASE3_REPORT_ERRATA.md`, and this draft uses the correct terms.
5. **Step 4B.** Test F1 was printed at the end of each training run and during evaluation, as pre-specified; nothing was changed after
   it. The pre-registration's text says "33" evaluation conditions but lists 34; all 34 were evaluated. No run failed or was excluded.
6. **Step 5S.** No deviation. The validation smoke test's output and log were committed and hashed in the pre-registration, but the
   script that produced them was not committed, so the smoke test cannot be re-run from the repository.
7. **Step 5.** No deviation from the registered evaluation and analysis. Before any composite was built, the scene pools were
   redesigned: excluding every scene used for earlier masks left no partly cloudy scene, so those scenes were kept and flagged, and
   arm C (the only arm trained on catalogue data) is reported as flagged. The composite build was interrupted twice and resumed; 89
   files from the interrupted runs were kept after verification by mask, shape, data type and value range, not byte for byte. A
   detector-input export check compared a float32 cloud fraction with the float64 manifest value at a 1e-12 tolerance and failed;
   it was changed to compare in float64, which all 126 fractions match exactly. All three are recorded in the pre-registration
   before it was committed. The rb99 operating point did not reach its recall target and uses the pre-specified fallback.

## References

[1] Hafner, S., Nascetti, A., Azizpour, H., & Ban, Y. (2022). Sentinel-1 and Sentinel-2 data fusion for urban change detection using a dual stream U-Net. *IEEE Geoscience and Remote Sensing Letters, 19*, 4019907. https://doi.org/10.1109/LGRS.2021.3119856

[2] Daudt, R. C., Le Saux, B., & Boulch, A. (2018). Fully convolutional siamese networks for change detection. In *Proc. IEEE ICIP*, 4063–4067. https://doi.org/10.1109/ICIP.2018.8451652

[3] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. In *MICCAI*, LNCS 9351, 234–241. https://doi.org/10.1007/978-3-319-24574-4_28

[6] Shafique, A., Cao, G., Khan, Z., Asad, M., & Aslam, M. (2022). Deep learning-based change detection in remote sensing images: A review. *Remote Sensing, 14*(4), 871. https://doi.org/10.3390/rs14040871

[13] Daudt, R. C., Le Saux, B., Boulch, A., & Gousseau, Y. (2018). Urban change detection for multispectral earth observation using convolutional neural networks. In *Proc. IEEE IGARSS*, 2115–2118. https://doi.org/10.1109/IGARSS.2018.8518015

[16] Bouthillier, X., Delaunay, P., Bronzi, M., Trofimov, A., Nichyporuk, B., Szeto, J., et al. (2021). Accounting for variance in machine learning benchmarks. In *Proceedings of Machine Learning and Systems (MLSys), 3*.

[17] Y. M., M. (2026). What does a missing post-change optical image cost? A multi-seed study of SAR–optical urban change detection on OSCD. Manuscript (`paper/RESEARCH_PAPER_FINAL.md`).

[18] Neverova, N., Wolf, C., Taylor, G. W., & Nebout, F. (2016). ModDrop: Adaptive multi-modal gesture recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 38*(8), 1692–1706. arXiv:1501.00102. [verify volume and pages on IEEE Xplore]

[19] Hafner, S., & Ban, Y. (2023). Multi-modal deep learning for multi-temporal urban mapping with a partly missing optical modality. In *Proc. IEEE International Geoscience and Remote Sensing Symposium (IGARSS)*, 6843–6846. arXiv:2306.00640. [add DOI]

[20] Francis, A., Mrziglod, J., Sidiropoulos, P., & Muller, J.-P. (2020). Sentinel-2 Cloud Mask Catalogue (Version 1). Zenodo. https://doi.org/10.5281/zenodo.4172871

[21] Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics, 6*(2), 65–70.

[22] Sentinel Hub (2017). Improving cloud detection with machine learning. Sentinel Hub blog (s2cloudless). [add URL]

[23] Skakun, S., et al. (2022). Cloud Mask Intercomparison eXercise (CMIX): An evaluation of cloud masking algorithms for Landsat 8 and Sentinel-2. *Remote Sensing of Environment, 274*, 112990. https://doi.org/10.1016/j.rse.2022.112990 [verify full author list, title and volume]

Reference numbers follow the companion manuscript [17] where shared; unused numbers are omitted.
