# Clouds Are Not Missing Data: A Pre-Registered Robustness Study of SAR–Optical Urban Change Detection on OSCD

**Manish Y. M.**
*[Institution]*
manishym946@gmail.com

*Draft, 2026-10-22. Every number in this draft is taken from the audited study reports in `experiments/robust/`
(`REPORT.md`, `PHASE3_REPORT.md`, `PHASE4A_REPORT.md`, `PHASE4ABIS_REPORT.md`, `PHASE4B_REPORT.md`) and checked against them by
`paper/check_robustness_paper_numbers.py`. Items marked [verify] need checking before submission.*

---

## Abstract

SAR–optical change detectors are usually trained and tested with all images present, yet in practice the post-change
optical image is often partly cloudy rather than absent. We study this gap with the dual-stream U-Net (DS-UNet) of Hafner et al. on the
OSCD urban change benchmark (Sentinel-1 VV and Sentinel-2), in five pre-registered steps, each with 10 paired training seeds,
a frozen validation-only selection protocol and an independent audit of every reported number.
(1) Whole-modality dropout training (B) does not cost F1 on full input (Δ = −0.003, 95% CI [−0.037, +0.031]) and improves F1
with the post-change optical image zero-filled by +0.241 relative to full-input training (A).
(2) Under simulated opaque clouds with real cloud shapes, both models fail: 10% cloud cover reduces B's F1 from 0.553 to 0.329,
and at 30% cover B's F1 is 0.191, close to the 0.098 of a detector that labels everything as change. Cloud-valued pixels are
read as change (false-alarm rate under the cloud 0.96–1.00).
(3) Zero-filling the cloud pixels with an oracle mask restores B to 0.539 / 0.516 / 0.499 F1 at 10 / 30 / 60% cover,
recovering 87–94% of the loss; the full-input model A recovers only 16–40%.
(4) With imperfect masks, over-masking is almost free (−0.003 to −0.010 F1 at 30% cover for 2–8 px dilation), while
under-masking is expensive (−0.090 to −0.250 for missed clouds, −0.100 to −0.243 for eroded edges).
(5) Training with cloud-shaped spatial zero-fill (C) gives no detectable difference from whole-modality dropout on any of the 7
primary conditions (all |Δ| < 0.02 F1, Holm p ≥ 0.55), and does not reduce the under-masking failure.
The results point to the cloud mask, not the fusion architecture, as the limiting factor in this setting: a recall-biased
cloud mask plus modality-dropout training is the cheapest effective remedy we found. All corruption is simulated, the 10 seeds
are training replicates on the same 10 test cities, and none of these results establishes robustness to real cloudy acquisitions.

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
et al. [1], and asks four questions in sequence, each pre-registered before its test evaluation:

- **Q1 (Phase 2):** does whole-modality dropout training make the model robust to a missing post-change optical image, and what does it cost on full input?
- **Q2 (Phase 3):** does that robustness carry over to a post-change image that is partly covered by (simulated) opaque cloud?
- **Q3 (Steps 4A and 4A-bis):** does zero-filling the cloud pixels at inference help, with an oracle mask and with realistic mask errors?
- **Q4 (Step 4B):** does training with cloud-shaped spatial zero-fill improve on whole-modality dropout?

The answers are, in order: yes, at no detectable full-input cost; no; yes with a good mask, and the mask must err toward
over-masking; and no detectable improvement. The practical conclusion is that, in this setting, a new architecture is not yet
warranted. The bottleneck is whether cloud-valued pixels reach the model.

Because OSCD is small (24 cities), we report every result with 10 paired seeds, 95% confidence intervals and Holm-corrected
tests, select every checkpoint and threshold on validation cities only, and treat the 10 seeds explicitly as training replicates
on one fixed test set, not as independent geographic samples. Every reported number was re-derived from raw confusion counts by an
independent audit script.

## 2. Related Work

**SAR–optical change detection.** Hafner et al. [1] fuse Sentinel-1 and Sentinel-2 in a dual-stream U-Net with late fusion and
report urban change detection on OSCD [13]. Siamese fully convolutional networks [2] and U-Net variants [3] are common
optical baselines; reviews [6] cover the broader field.

**Missing modalities.** Modality dropout, randomly removing an input modality during training, was introduced for multimodal
gesture recognition [18] and is widely used to make fusion models usable when one input is absent. For urban mapping with
missing optical data, Hafner and Ban proposed reconstructing optical features from SAR [19, verify]; we do not compare with that
method here.

**Clouds.** Real cloud masks for Sentinel-2 are available in the Sentinel-2 Cloud Mask Catalogue [20], which we use for cloud
shapes. Operational cloud detectors (e.g. scene classification layers or s2cloudless [verify citation]) are imperfect, missing
thin cloud and cloud edges; we model their errors only synthetically.

**Evaluation practice.** Seed variance can reverse model rankings in small benchmarks [16, 17]. We follow a pre-registration
discipline: each step's hypotheses, conditions, test family and decision rule were written down and hashed before any test output
of that step existed.

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
2500: 0.405; mean of 3 seeds). Training is bitwise deterministic on our hardware (PyTorch 2.5.1+cu124, cuDNN deterministic,
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
validation or test mask.

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

Zero is the fill value used in these experiments. It is not an intrinsic missing-data token: the models treat it as missing only
to the extent that training showed them zero-filled inputs.

### 3.5 Statistics and integrity

F1, precision, recall, IoU and false-alarm rate are computed from pooled pixel counts over the 10 test cities. Each step has one
primary family of paired comparisons over seeds 1–10 (two-sided paired t-test, Holm correction [21]), reported with the mean
difference, its 95% CI, d_z, the number of seeds favouring each arm, and a Holm-corrected Wilcoxon signed-rank test (smallest
attainable p with n = 10 is 0.0020). Secondary analyses form their own Holm families and are labelled descriptive. With 10 pairs,
the minimum detectable effect at 80% power is 1.00 SD of the paired differences at α = 0.05 and 1.41 SD at α = 0.05/7,
roughly 0.04–0.07 F1 here; non-significant results are not evidence of equivalence.

Each step was pre-registered (hypotheses, conditions, primary family, decision guidance, and sha256 hashes of all code and
inputs) before its first test evaluation. Each evaluation enforces anchors, for example that re-evaluated clean counts equal the
training-time counts exactly. An independent audit script recomputes every reported statistic from raw counts (Phase 2: 579 checks;
Phase 3: 167; Step 4A-bis: 1338; Step 4B: 432; all with 0 failures). Before Step 4B training, a parity run of the spatial-dropout
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

### 4.3 Zero-filling under the cloud recovers most of the loss, for the dropout model only (Step 4A)

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
rate inside the mask ≈ 0.51). The remedy works only together with dropout training. B's remaining gap is lost evidence rather than false
alarms: its recall inside the mask falls from about 0.73 to 0.55–0.61 when it must rely on SAR and the pre-change image.

### 4.4 The mask must err toward over-masking (Step 4A-bis)

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

## 5. Discussion

**Missing and cloudy are different problems.** Phase 2 and Phase 3 together show that robustness to a missing image does not
transfer to a partly corrupted one. A dropout-trained model learns that a zero-filled optical image carries no information. It
does not learn that a bright, flat, cloud-valued patch is not a change. In this model, cloud-valued pixels act as strong evidence
of change, both under the cloud and, through the receptive field, in clear pixels nearby.

**The remedy is in the input, not in the architecture.** Zero-filling cloud pixels turns the cloud problem back into the missing-data
problem that dropout training already solves. With a perfect mask this recovers 87–94% of the cloud loss. With an imperfect mask, the
outcome depends almost entirely on the direction of the error: a mask that covers 21% of the clear scene in addition to the cloud
costs B only 0.010 F1, while a mask that misses a quarter of the cloud costs 0.168. This asymmetry is a design requirement for the
cloud detector: it should be tuned for recall (e.g. dilated) rather than for balanced accuracy. That requirement holds only for a
model trained with dropout; the full-input model reads zero-filled clear pixels as change, so over-masking hurts it.

**Training-time augmentation has reached its limit here.** Spatial dropout teaches the model about zero-filled regions with cloud
shapes, but B already handles zero-filled regions well, and the residual failure is cloud values that the mask lets through. C never
sees cloud values in training, so the null result is what the Step 4B pre-registration expected. Closing the remaining gap requires the
model to see or be told about cloud-valued pixels, for example training that includes cloud-valued corruption, or an explicit
per-pixel validity channel.

**Architecture decision.** Under the pre-specified decision guidance, these results do not justify a new fusion architecture now.
The next steps, each to be pre-registered separately, are (i) a recall-biased cloud detector evaluated with B, preferably on real cloudy
acquisitions rather than simulated ones, and (ii) training with cloud-valued corruption or a validity-mask input channel. The validity channel
is the minimal architecture change and should be compared against (i) under the same protocol. A reliability-gated fusion architecture
would be justified only if both leave a specific, measured failure.

**Operating point.** Dropout-trained models shift their output probabilities upward and need their decision threshold chosen on
validation data; at a fixed 0.5 threshold, B loses 0.086 F1 on full input. Thresholds also vary widely between seeds (C's frozen
clean thresholds range from 0.50 to 0.97). Deployments should calibrate the threshold on held-out data from the target region.

## 6. Limitations

- **Simulated corruption.** The cloud shapes are real, but the clouds are opaque, uniform, hard-edged and shadowless, built from one fixed
  spectrum, and only the post-change date is corrupted. This is not equivalent to real cloudy acquisitions, which include thin cloud,
  haze, cloud shadow and atmospheric effects. The haze condition (Phase 3, secondary) is the only partial-transparency test.
- **Synthetic mask errors.** Erosion, dilation and whole-cloud removal do not model a specific detector. Real detectors also miss thin
  cirrus and shadows and mistake bright surfaces for cloud.
- **Zero as a fill value.** Zero is the fill value used here, applied to [0, 1]-scaled reflectance. It is not an intrinsic missing-data token.
- **Seeds are not geography.** The 10 seeds are training replicates evaluated on the same 10 OSCD test cities. They measure training
  variability; the p-values do not support claims about other regions, sensors or seasons.
- **Small benchmark, limited power.** OSCD has 24 cities; checkpoints and thresholds are selected on 4 validation cities. Effects below
  about 0.04–0.07 F1 can be missed in the Step 4B comparison.
- **Confounded Step 4B contrast.** C differs from B both by adding spatial dropout and by halving whole-image dropout.
- **One architecture, one dropout mixture per arm.** The mixtures were fixed a priori and not tuned. The results may differ for other
  fusion designs (e.g. early fusion or attention-based models).
- **Baseline gap.** Our reproduction of the official DS-UNet protocol (0.557 ± 0.035) is below the published 0.600, and two small code
  differences from the official training loop remain (batch sampling with replacement; first-orbit SAR at evaluation).
- **No comparison with SAR-to-optical reconstruction** methods [19].

## 7. Conclusion

In a five-step, pre-registered study of SAR–optical urban change detection on OSCD with 10 paired seeds per arm, we found that
(1) whole-modality dropout makes a DS-UNet robust to a missing post-change optical image at no detectable full-input cost;
(2) that robustness does not extend to partly cloudy images, because cloud-valued pixels are read as change;
(3) zero-filling the cloud pixels restores most of the lost F1, but only for the dropout-trained model;
(4) the cloud mask must err toward over-masking, because unmasked cloud pixels are expensive and over-masked clear pixels are nearly free; and
(5) training with cloud-shaped spatial zero-fill adds no detectable robustness.
For practitioners, the cheapest effective configuration we found is a recall-biased cloud mask, zero-fill under the mask, and a
modality-dropout-trained model with a validation-calibrated threshold. Whether this holds on real cloudy acquisitions is the most
important open question.

## Reproducibility statement

Code, pre-registrations, run files, logs, analysis tables, audits and reports are in the repository (`experiments/robust/`,
branch `robust-phase4`, commits `eead015`, `25fe13b`, `5dfaa17`). Checkpoints and raw count arrays are excluded from version control and
listed with sha256 hashes in `experiments/robust/ARTIFACT_SHA256_2026-10-22_post4b.txt`. Each pre-registration's hash table can be
re-verified with `verify_prereg_hashes.py`. Training is bitwise deterministic on the hardware used.

## Appendix A: Deviations and gaps (as recorded in the step reports)

1. **Phase 2 step budget.** A 1000-step budget was first proposed after the test F1 of a 3-seed step sweep had been viewed. That choice
   was discarded and the budget re-chosen on validation score (2500 steps). Seeds 1–3 of A share the configuration of a sweep run whose
   test F1 had been seen; they were retrained from scratch, reproduced identically, and no Phase 2 decision used them.
2. **Phase 3 masks.** The first mask build used an absolute cloud-fraction tolerance (±0.05), which could nearly double a small cloud.
   The masks were rebuilt with a relative tolerance before any model saw a mask; the superseded masks are kept and unused. A missing
   `model.eval()` in the Phase 3 evaluation script was fixed after a validation-only smoke test, before any test evaluation.
3. **Step 4A-bis.** The pre-registration cites a validation-only smoke test for which no log exists; that claim is unverified. The
   evaluation printed per-seed test F1 while running; the analysis script's hash equals the registered hash. The analysis crashed on
   its final console print and was re-run unchanged. Two source files were hashed in every run file but missing from the pre-registration's
   hash table; their hashes are unchanged since Phase 3. Inside/outside-mask metrics and IoU were added as a labelled exploratory addendum.
4. **Phase 3 report wording.** `PHASE3_REPORT.md` describes the all-change F1 (≈ 0.098) as "the change prevalence". The prevalence
   is 0.052; 0.098 is the F1 of the all-change detector. The report is left unchanged; this draft uses the correct terms.
5. **Step 4B.** Test F1 was printed at the end of each training run and during evaluation, as pre-specified; nothing was changed after
   it. The pre-registration's text says "33" evaluation conditions but lists 34; all 34 were evaluated. No run failed or was excluded.

## References

[1] Hafner, S., Nascetti, A., Azizpour, H., & Ban, Y. (2022). Sentinel-1 and Sentinel-2 data fusion for urban change detection using a dual stream U-Net. *IEEE Geoscience and Remote Sensing Letters, 19*, 4019907. https://doi.org/10.1109/LGRS.2021.3119856

[2] Daudt, R. C., Le Saux, B., & Boulch, A. (2018). Fully convolutional siamese networks for change detection. In *Proc. IEEE ICIP*, 4063–4067. https://doi.org/10.1109/ICIP.2018.8451652

[3] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. In *MICCAI*, LNCS 9351, 234–241. https://doi.org/10.1007/978-3-319-24574-4_28

[6] Shafique, A., Cao, G., Khan, Z., Asad, M., & Aslam, M. (2022). Deep learning-based change detection in remote sensing images: A review. *Remote Sensing, 14*(4), 871. https://doi.org/10.3390/rs14040871

[13] Daudt, R. C., Le Saux, B., Boulch, A., & Gousseau, Y. (2018). Urban change detection for multispectral earth observation using convolutional neural networks. In *Proc. IEEE IGARSS*, 2115–2118. https://doi.org/10.1109/IGARSS.2018.8518015

[16] Bouthillier, X., Delaunay, P., Bronzi, M., Trofimov, A., Nichyporuk, B., Szeto, J., et al. (2021). Accounting for variance in machine learning benchmarks. In *Proceedings of Machine Learning and Systems (MLSys), 3*.

[17] Y. M., M. (2026). What does a missing post-change optical image cost? A multi-seed study of SAR–optical urban change detection on OSCD. Manuscript (`paper/RESEARCH_PAPER_FINAL.md`).

[18] Neverova, N., Wolf, C., Taylor, G., & Nebout, F. (2016). ModDrop: Adaptive multi-modal gesture recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 38*(8), 1692–1706. [verify]

[19] Hafner, S., & Ban, Y. (2023). Multi-modal deep learning for multi-temporal urban mapping with a partly missing optical modality. [verify title, venue and DOI]

[20] Francis, A., Mrziglod, J., Sidiropoulos, P., & Muller, J.-P. (2020). Sentinel-2 Cloud Mask Catalogue (Version 1). Zenodo. https://doi.org/10.5281/zenodo.4172871

[21] Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics, 6*(2), 65–70.

Reference numbers follow the companion manuscript [17] where shared; unused numbers are omitted.
