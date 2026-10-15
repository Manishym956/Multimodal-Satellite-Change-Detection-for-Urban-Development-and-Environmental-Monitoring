# Results (v2)

> **Version note:** This is v2, incorporating corrections from the scientific consistency
> audit (October 2026). Changes from v1 are marked `[CORRECTED]`.
> See `03_evidence_corrections_v2.md` for the full audit log.

All metrics in this section are computed from saved confusion counts (TP, FP, FN, TN) at
decision threshold τ = 0.50 unless otherwise stated. Per-city and pooled totals are taken
from the corresponding `metrics.json` files; no values are interpolated or estimated.
Recomputed rates match stored values to < 1 × 10⁻⁷ in all cases.

Notation used throughout:

- **All-pixel** — denominator includes every pixel in the test image, including pixels
  where SAR nodata was filled with zeros.
- **Valid-pixel** — denominator restricted to pixels where $M(i,j) = 1$, meaning neither
  VV nor VH contained NaN in the original raster at either time step. Validity is recorded
  from pre-fill NaN indicators; genuinely zero-valued (non-NaN) SAR pixels are treated as
  valid. **[CORRECTED: v1 described valid-pixel as "SAR data is non-zero in both time
  steps." The actual criterion is based on original NaN indicators.
  Source: `src/datasets/scenario2_dataset.py`, line 113.]**
- **F1** — harmonic mean of precision and recall.
- **IoU** — Intersection over Union (Jaccard index) for the change class.
- **FAR** — False-Alarm Rate = FP / (FP + TN).

---

## 3.1  OSCD Benchmark

### Dataset and evaluation protocol

The OSCD benchmark uses 24 Sentinel-2 cities split 10 / 4 / 10 (train / val / test).
All models are evaluated on the 10 held-out test cities with pooled confusion counts
(pixels summed across cities before computing rates). The reference labels are binary
change/no-change derived from the OSCD hand-annotations.

SAR data on OSCD is Sentinel-1 multi-temporal composite imagery. Exact acquisition
timestamps for the composite are not available from the dataset metadata; therefore no
precise temporal-gap characterisation is reported. RQ2 (temporal gap sensitivity) and
RQ3 (SAR polarisation contribution) are not addressed by OSCD experiments.

### Architecture and loss overview

CNN-family models (SAR-only U-Net, Optical-only U-Net, Sync Early Fusion U-Net,
Sync Dual-Stream CNN, Naive Async CNN, Clean Async CNN) were trained with standard
unweighted BCE (pos_weight = 1; `pos_weight` key absent in all CNN config files).
Transformer-family models (Self-Attention, Cross-Attention) were trained with
class-weighted BCE (pos_weight = 41.0), matching the approximate 1 : 40.7 pixel-level
class ratio in the training set.

**This loss difference confounds any direct architectural comparison between the CNN and
Transformer families. Cross-family F1 comparisons reflect both architecture and loss
function; they do not isolate either factor.**

### Table 1 — OSCD benchmark results (τ = 0.50, 10 test cities pooled)

| Model | Params | Loss | Best Ep | Val F1 | Test Prec | Test Rec | Test F1 | Test IoU | Test FAR | TP | FP | FN | TN |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAR-only U-Net | 31,037,057 | BCE w=1 | 21 | 0.0662 | 0.4558 | 0.2917 | 0.3558 | 0.2164 | 0.0190 | 46,407 | 55,401 | 112,670 | 2,863,458 |
| Optical-only U-Net | 31,050,881 | BCE w=1 | 50 | 0.2387 | 0.2515 | 0.2830 | 0.2663 | 0.1536 | 0.0459 | 45,025 | 134,013 | 114,052 | 2,784,846 |
| Sync Early Fusion U-Net | 31,052,033 | BCE w=1 | 46 | 0.2538 | 0.2877 | 0.4533 | 0.3520 | 0.2136 | 0.0612 | 72,104 | 178,489 | 86,973 | 2,740,370 |
| Sync Dual-Stream CNN | 27,284,097 | BCE w=1 | 31 | 0.3523 | 0.3009 | 0.5415 | 0.3868 | 0.2398 | 0.0686 | 86,137 | 200,124 | 72,940 | 2,718,735 |
| Naive Async CNN | 27,284,097 | BCE w=1 | 46 | 0.1093 | 0.2363 | 0.1814 | 0.2052 | 0.1143 | 0.0319 | 28,849 | 93,242 | 130,228 | 2,825,617 |
| Clean Async CNN | 27,276,609 | BCE w=1 | 26 | 0.0992 | 0.4753 | 0.3631 | 0.4117 | 0.2592 | 0.0218 | 57,764 | 63,762 | 101,313 | 2,855,097 |
| Transformer Self-Attn | 1,358,081 | BCE w=41 | 37 | 0.1333 | 0.0974 | 0.4734 | 0.1615 | 0.0879 | 0.2391 | 75,307 | 697,994 | 83,770 | 2,220,865 |
| Transformer Cross-Attn | 1,156,993 | BCE w=41 | 33 | 0.1428 | 0.2615 | 0.5592 | 0.3564 | 0.2168 | 0.0861 | 88,959 | 251,196 | 70,118 | 2,667,663 |

*Parameter counts verified from checkpoint state_dict key-shape products.*
*All-pixel protocol. SAR on OSCD is multi-temporal composite; no-data masking is not applied.*
*Best epoch selected by validation F1 (all-pixel) in all OSCD runs. Every OSCD row is seed 42. The Clean Async CNN test F1 of 0.4117 is that single-seed observation. It is not conclusive and it does not generalize to Scenario 2 or to other seeds.*

### Observations

**SAR-only U-Net (F1 = 0.356, FAR = 0.019).**
The SAR-only baseline achieves the highest precision of the U-Net family (0.456) with the
lowest false-alarm rate (0.019), indicating that when it predicts change, it is relatively
often correct. Recall is 0.292, indicating that many change pixels are missed.

**Optical-only U-Net (F1 = 0.266, FAR = 0.046).**
Replacing SAR with optical imagery (both time steps) does not improve F1 over the
SAR-only model. Recall (0.283) is slightly lower than the SAR-only model (0.292).
**[CORRECTED: v1 stated "Recall improves slightly (0.283 vs. 0.292)." This is backwards;
0.283 < 0.292. Optical-only recall is lower, not higher.]** Precision also drops
substantially (0.252 vs. 0.456). The lower F1 of the optical-only model relative to the
SAR-only model is specific to this OSCD partition and does not generalise across datasets
or conditions. Contributing factors such as optical data quality, label alignment, and
spectral-change noise are not separable in this design.

**Sync Early Fusion U-Net (F1 = 0.352, FAR = 0.061) and Sync Dual-Stream CNN (F1 = 0.387, FAR = 0.069).**
Synchronous multimodal fusion models achieve higher recall (0.453 and 0.542) at the cost
of higher false-alarm rates compared to the SAR-only and optical-only unimodal models.
The Dual-Stream CNN produces higher F1 and IoU than the Early Fusion U-Net, with best-epoch
selection at epoch 31 vs. 46.

**Naive Async CNN (F1 = 0.205, 27,284,097 params) vs. Clean Async CNN (F1 = 0.412, 27,276,609 params).**
These two models share the same Dual-Stream U-Net architecture but differ in how the
withheld $T_2$ optical data is handled and consequently in their total parameter count.
The Naive model uses 28 input channels: 2 SAR + 13 optical $T_1$ + 13 zero-filled
$T_2$ placeholders. The Clean model uses 15 input channels: 2 SAR + 13 optical $T_1$
(no $T_2$ placeholder channels). The difference in input width changes the first encoder
layer weight tensor, yielding a parameter difference of 7,488 (27,284,097 vs. 27,276,609).

**[CORRECTED: v1 stated "Both models have the same architecture and parameter count
(27,284,097)." This is incorrect; their parameter counts differ by 7,488 due to the
different optical encoder input widths. Source: `src/models/dual_stream_unet_clean.py`,
docstring; `results/tables/experiment_summary.json`.]**

The observed F1 difference (0.412 vs. 0.205) is a single seed (seed 42). It is associated with the combination of
these two changes: the presence of zero-filled input channels and the resulting difference
in first-layer weights. **These two factors are not separable in this design; the
comparison is between two input configurations rather than a controlled ablation of
zero-filling alone.** The 0.4117 test F1 is not a conclusive or generalizable estimate of clean asynchronous performance.

**Transformer models (Self-Attn F1 = 0.162, Cross-Attn F1 = 0.356).**
The Self-Attention Transformer has the highest recall in the benchmark (0.473) but also
the highest false-alarm rate (0.239), resulting in low precision (0.097) and low F1
(0.162). The Cross-Attention Transformer achieves F1 = 0.356 and FAR = 0.086, numerically
comparable to the Sync Early Fusion U-Net despite having approximately 27× fewer parameters
(1.16 M vs. 31.05 M). Both Transformer models used pos_weight = 41; the CNN models used
unweighted BCE (w=1). This difference in loss configuration confounds any inference about
architectural efficiency or capacity.

**Validation–test discrepancy.**
For several models the test-set F1 at τ = 0.50 differs materially from the validation F1
used for checkpoint selection. For example, the SAR-only U-Net achieves val F1 = 0.066
but test F1 = 0.356; the Sync Dual-Stream CNN achieves val F1 = 0.352 and test F1 = 0.387.
This discrepancy is observed but not causally attributed: possible contributing factors
include distribution differences between the 4 validation cities and the 10 test cities,
and threshold sensitivity differences across splits.

---

## 3.2  Scenario 2 — SpaceNet 7 Initial Baselines (Seed 42)

### Dataset and evaluation protocol

SpaceNet 7 provides monthly Sentinel-2 mosaics (10 spectral bands) and co-located
Sentinel-1 dual-polarisation SAR (VV, VH), giving 4 SAR channels per time step. The
locked geographic split assigns 20 / 4 / 4 sites to train / val / test. This split is
fixed by SHA-256:
`5d0841adf0c651166cb18ecea310c6ef8cce69172065fc2d699a29083c272e27` and is not modified
for any subsequent experiment.

Change labels are building-derived proxy annotations (XOR of thresholded building
probability maps at $T_1$ and $T_2$); they are not independently validated field
annotations.

**Loss configurations for Scenario 2:**
Both the standard and masked loss conditions use `pos_weight = 94.0`, matching the
training set class ratio (50,387 change / 4,718,348 no-change ≈ 1 : 93.64).

- **Standard condition:** `nn.BCEWithLogitsLoss(pos_weight=94.0)`. All pixels in the
  batch contribute to the loss numerator; the denominator is the total pixel count.
- **Masked condition:** `MaskedBCEWithLogitsLoss(pos_weight=94.0)`. Only valid-SAR pixels
  ($M=1$) contribute to the numerator; the denominator is $\sum M$ (valid pixel count).

**[CORRECTED: v1 described the standard condition as "unweighted BCE." Both conditions
use pos_weight=94.0. Source: `train_scenario2_dual_stream.py` line 345;
`train_scenario2_sync_multiseed.py` lines 317–322.]**

**Checkpoint selection — initial baselines (seed 42).**
The three standard-loss baselines were selected by validation F1 (all-pixel).
The three masked-loss runs were selected by validation F1 (valid-pixel).
This difference in selection criterion is a confound when comparing the two loss
conditions on the seed-42 results; it is corrected in the matched multi-seed experiment.

### Table 2 — Scenario 2 initial baselines (seed 42, τ = 0.50)

| Model | Loss | Sel. Ep | Val F1 | Prec (all) | Rec (all) | F1 (all) | FAR (all) | F1 (valid) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Async CNN | std weighted BCE w=94 | 2 | 0.1832 (all) | 0.0479 | 0.2334 | 0.0794 | 0.0336 | 0.0804 |
| Sync CNN | std weighted BCE w=94 | 21 | 0.2663 (all) | 0.0610 | 0.5680 | 0.1102 | 0.0633 | 0.1107 |
| Cross-Attn | std weighted BCE w=94 | 23 | 0.1328 (all) | 0.0214 | 0.5705 | 0.0412 | 0.1892 | 0.0411 |
| Async CNN | masked weighted BCE w=94 | 2 | 0.1677 (valid) | 0.0473 | 0.2907 | 0.0813 | 0.0424 | 0.0822 |
| Sync CNN | masked weighted BCE w=94 | 30 | 0.2878 (valid) | 0.0781 | 0.5342 | 0.1363 | 0.0456 | 0.1376 |
| Cross-Attn | masked weighted BCE w=94 | 41 | 0.2223 (valid) | 0.0267 | 0.4639 | 0.0506 | 0.1222 | 0.0506 |

*Val F1 column indicates which protocol was used for checkpoint selection.*
*F1 (valid) for the three std-loss runs is derived from post-hoc ablation evaluation*
*(scratch/ablation_eval_results.json); these runs did not log valid-pixel metrics during training.*
*All Scenario 2 models: 20 train sites, 4 val sites, 4 test sites (locked split).*

### Observations

**Overall performance range.**
All Scenario 2 models achieve all-pixel F1 in the range 0.04–0.14 at τ = 0.50.
The OSCD and Scenario 2 benchmarks differ in label type, geographic scope, class
imbalance, sensor configuration, and evaluation conditions. Their F1 scores are not
directly comparable. **[CORRECTED: v1 stated Scenario 2 "presents a harder task" and
listed reasons. This was an unsupported cross-dataset difficulty claim. The lower
Scenario 2 F1 is consistent with the dataset differences but does not establish
inherent task difficulty.]**

**Async CNN baseline (seed 42, best epoch 2).**
The Async CNN baseline selects its checkpoint at epoch 2, the earliest epoch allowed.
This early selection is observed in the training history; possible explanations include
training instability, rapid overfitting, or the particular validation dynamics under the
all-pixel selection criterion. The all-pixel F1 of 0.079 and valid-pixel F1 of 0.080
are the lowest among the three architectures.

**Sync CNN: standard vs. masked (seed 42, uncontrolled).**
The Sync CNN masked-loss variant (valid-pixel F1 = 0.1376) and the std-loss baseline
(valid-pixel F1 = 0.1107) differ in both loss function and checkpoint-selection criterion
(all-pixel for std-loss, valid-pixel for masked-loss). These two differences are
confounded; the F1 difference may partly reflect the selection protocol rather than the
loss function. The controlled comparison in Section 3.3 addresses this.

**Cross-Attention model.**
The Cross-Attention Transformer achieves high recall (0.57 for both conditions) but the
highest FAR of the three architectures (0.19 for std-loss, 0.12 for masked-loss),
resulting in low precision and F1. The masked-loss variant is associated with a lower
FAR (0.19 → 0.12) while maintaining similar recall under the selected checkpoints.
Because the selection criteria differ between the two conditions, this FAR difference
cannot be attributed to the loss function alone.

---

## 3.3  Matched Multi-Seed Experiment — Sync CNN

### Design

To isolate the effect of the loss function from the confounds present in the seed-42
comparison, a controlled multi-seed experiment was conducted using only the Sync
Dual-Stream CNN. Four additional runs were created: seeds 43 and 44, each with
standard-weighted-BCE and masked-weighted-BCE.

**Controls held constant across both loss conditions within each seed:**
- Initial model weights (identical state_dict for each seed pair, confirmed by preflight
  state_dict comparison).
- Training crop coordinates and crop order (deterministic pre-generated crop schedule
  using a separate `numpy.random.RandomState(seed)` RNG, consumed identically by both
  conditions for the same seed).
- Checkpoint-selection criterion: validation F1 (valid-pixel) for all four runs.
- Training sites, validation sites, preprocessing, batch size, Adam lr = 1 × 10⁻⁴,
  pos_weight = 94.0, 50 epochs, crop size 32 × 32, 64 crops per site per epoch.

Seeds 43 and 44 use different initializations and crop schedules from each other and from
seed 42.

**Relationship to seed 42.**
The seed-42 Sync CNN masked-BCE run used valid-pixel checkpoint selection and is included
in Table 3 for context. The seed-42 std-BCE run used all-pixel checkpoint selection and
is not directly comparable under the controlled protocol; it is listed in Table 3 with a
dagger (†) note.

### Table 3 — Matched multi-seed Sync CNN results (τ = 0.50)

| Seed | Loss | Peak Ep | Val F1 (valid) | F1 (all) | F1 (valid) |
|---|---|---:|---:|---:|---:|
| 42† | std weighted BCE w=94 | 21 | 0.2663 (all)† | 0.1102 | 0.1107 |
| 42 | masked weighted BCE w=94 | 30 | 0.2878 | 0.1363 | 0.1376 |
| 43 | std weighted BCE w=94 | 30 | 0.2932 | 0.1327 | 0.1342 |
| 43 | masked weighted BCE w=94 | 30 | 0.2877 | 0.1236 | 0.1257 |
| 44 | std weighted BCE w=94 | 43 | 0.2746 | 0.1327 | 0.1337 |
| 44 | masked weighted BCE w=94 | 43 | 0.2744 | 0.1141 | 0.1180 |

*† Seed-42 std-BCE run used all-pixel checkpoint selection; it is not part of the matched comparison.*
*Seeds 43 and 44 use valid-pixel checkpoint selection for both loss conditions.*
*Peak epoch is the epoch selected by the checkpoint-selection criterion.*

**Confusion counts (τ = 0.50) — verified from metrics.json:**

| Seed | Loss | TP | FP | FN | TN |
|---|---|---:|---:|---:|---:|
| 43 | std weighted BCE | 3,669 | 44,770 | 3,190 | 902,899 |
| 43 | masked weighted BCE | 3,923 | 52,695 | 2,936 | 894,974 |
| 44 | std weighted BCE | 3,924 | 48,353 | 2,935 | 899,316 |
| 44 | masked weighted BCE | 3,822 | 56,318 | 3,037 | 891,351 |

*Sources: `experiments/scenario_2_multiseed/sync_cnn_{baseline,masked}_seed{43,44}/metrics.json`,*
*key: `test_threshold_0_5.all_pixel_protocol`*

### Observations

**Peak epoch alignment.**
For both seeds 43 and 44, the std-loss and masked-loss conditions reach peak valid-pixel
validation F1 at the same epoch (seed 43: epoch 30 for both; seed 44: epoch 43 for both).
This alignment suggests the training dynamics under identical crop schedules, initializations,
and selection criteria converge to the same peak epoch regardless of loss type, for these
two seeds.

**Within-seed F1 comparison.**
At τ = 0.50 under the matched protocol:
- Seed 43: std-loss F1 (all) = 0.1327 vs. masked-loss = 0.1236; difference = −0.0091.
- Seed 44: std-loss F1 (all) = 0.1327 vs. masked-loss = 0.1141; difference = −0.0186.

In both matched seeds, the masked-loss condition produces lower all-pixel F1 at τ = 0.50
than the std-loss condition. The confusion counts show that masked-loss conditions tend to
produce higher recall (fewer FN) but also substantially higher FP compared to std-loss
within the same seed.

**Descriptive summary across matched seeds (N = 2; descriptive only).**
Baseline all-pixel F1: mean = 0.1327, std = 0.0000.
Masked all-pixel F1: mean = 0.1188, std = 0.0067.
Mean paired difference (masked − baseline): −0.0139.
With N = 2 seeds, these observations are descriptive. No inference about the distribution
of this effect is warranted.

---

## 3.4  Secondary Operating Point — Calibrated Threshold τ*

For each run, a calibrated threshold τ* was identified by maximising F1 (valid-pixel) on
the pooled validation scenes. Table 4 reports results at both τ = 0.50 and the
corresponding τ* for all Sync CNN runs with calibrated-threshold results available.
These results are **secondary and descriptive**: τ* is selected per-run on the validation
set, so models are not compared at the same threshold. Differences between runs at their
respective τ* values do not constitute a controlled comparison.

### Table 4 — Sync CNN results at τ = 0.50 and calibrated τ* (valid-pixel protocol)

| Seed | Loss | τ* | F1 (all) @ 0.50 | F1 (valid) @ 0.50 | F1 (all) @ τ* | F1 (valid) @ τ* |
|---|---|---:|---:|---:|---:|---:|
| 42 | std weighted BCE | 0.50 | 0.1102 | 0.1107 | 0.1102 | 0.1107 |
| 42 | masked weighted BCE | 0.78 | 0.1363 | 0.1376 | 0.1909 | 0.1934 |
| 43 | std weighted BCE | 0.79 | 0.1327 | 0.1342 | 0.1943 | 0.1971 |
| 43 | masked weighted BCE | 0.83 | 0.1236 | 0.1257 | 0.1912 | 0.1944 |
| 44 | std weighted BCE | 0.86 | 0.1327 | 0.1337 | 0.2180 | 0.2191 |
| 44 | masked weighted BCE | 0.91 | 0.1141 | 0.1180 | 0.2113 | 0.2127 |

*τ* selected by maximising valid-pixel F1 on the validation set.*
*Seed-42 std-BCE τ* = 0.50 indicates the validation-optimal threshold coincides with 0.50.*
*Both All-Pixel and Valid-Pixel metrics reported at each threshold for completeness.*

**Observation.**
At calibrated τ*, the F1 gap between std-loss and masked-loss conditions is smaller in
magnitude than at τ = 0.50. For seed 43, std-loss at τ* achieves F1 (valid) = 0.1971 vs.
masked-loss = 0.1944 (difference = −0.0027, vs. −0.0085 at τ = 0.50). For seed 44,
std-loss at τ* achieves F1 (valid) = 0.2191 vs. masked-loss = 0.2127 (difference = −0.0064).

The masked-loss runs in the matched experiment select higher optimal thresholds
(τ* = 0.83 and 0.91) than the corresponding std-loss runs (τ* = 0.79 and 0.86). Whether
this pattern reflects a systematic shift in output distribution due to the masking
mechanism, or is a consequence of the particular validation dynamics at these two seeds,
cannot be determined from N = 2 observations. **[CORRECTED: v1 stated this was
"consistent with the effect of masking on the output distribution," which attributes a
mechanism. The observation is that masked-loss runs select higher τ* in both matched
seeds; the mechanism is not established.]**

---

## 3.5  Summary of Supported Findings

The following findings are directly supported by the saved experiment artifacts:

1. **Naive vs. Clean Async CNN (OSCD, seed 42 only).**
   The Naive Async CNN (28 channels, including 13 zero-filled T2 optical placeholders,
   27,284,097 params) achieves F1 = 0.205, compared to the Clean Async CNN (15 channels,
   no placeholders, 27,276,609 params) which achieves F1 = 0.4117, on the same 10 OSCD
   test cities under identical evaluation conditions. Both are CNNs trained with unweighted BCE.
   The two models differ in both the
   input configuration and the first-layer parameter count. **The observed F1 difference
   is a single-seed observation. It is associated with these combined differences; the experiment does not isolate
   zero-filling as the sole cause, and it does not generalize beyond this seed.** (Section 3.1)

2. **Cross-Attention Transformer vs. Early Fusion U-Net (OSCD).**
   The Cross-Attention Transformer achieves F1 = 0.356 with 1.16 M parameters, numerically
   comparable to the Sync Early Fusion U-Net (F1 = 0.352, 31.05 M parameters). The loss
   functions differ (BCE w=41 vs. BCE w=1), which confounds direct comparison. (Section 3.1)

3. **Scenario 2 Sync CNN seed-42 comparison (uncontrolled).**
   The masked-loss Sync CNN (seed 42) achieves valid-pixel F1 = 0.1376 vs. the std-loss
   baseline valid-pixel F1 = 0.1107. These runs differ in both loss function and
   checkpoint-selection criterion; the comparison is not controlled. (Section 3.2)

4. **Matched multi-seed comparison (seeds 43 and 44).**
   In the controlled matched comparison, the masked-loss condition does not improve
   all-pixel F1 over std-loss at τ = 0.50 for either seed (differences: −0.0091 and
   −0.0186, favouring std-loss in both cases). At calibrated τ*, the gap is smaller but
   std-loss remains at or above masked-loss in valid-pixel F1 for both seeds. (Section 3.3)

5. **τ* pattern.**
   The masked-loss runs in the matched experiment select higher optimal thresholds than
   the corresponding std-loss runs. Whether this reflects a mechanism or is a coincidence
   at N = 2 is not established. (Section 3.4)

The following are **not** supported by the available experiments:

- Any claim about cross-dataset task difficulty ordering (OSCD vs. Scenario 2).
- Any claim that masked-loss improves performance across architectures or datasets.
- Any claim that the Naive-vs-Clean F1 difference isolates zero-filling as the sole cause.
- Any claim that the above differences are statistically significant (N = 2 seeds).
- Any temporal-gap experiment or single-polarisation vs. dual-polarisation ablation result.
- Any result based on independently validated change annotations for Scenario 2
  (labels are building-derived proxies).
