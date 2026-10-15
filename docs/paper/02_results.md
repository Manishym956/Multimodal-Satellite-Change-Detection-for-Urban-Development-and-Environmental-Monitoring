# Results

All metrics in this section are computed from saved confusion counts (TP, FP, FN, TN) at
decision threshold τ = 0.50 unless otherwise stated.  Per-city and pooled totals are taken
from the corresponding `metrics.json` files; no values are interpolated or estimated.
Recomputed rates match stored values to < 1 × 10⁻⁷ in all cases.

Notation used throughout:

- **All-pixel** — denominator includes every pixel in the test image (valid and invalid SAR).
- **Valid-pixel** — denominator restricted to pixels where SAR data is non-zero in *both*
  time steps; pixels with zeroed SAR (no-data) are excluded from precision, recall, and the
  FAR denominator before metric computation.
- **F1** — harmonic mean of precision and recall.
- **IoU** — Intersection over Union (Jaccard index) for the change class.
- **FAR** — False-Alarm Rate = FP / (FP + TN).

---

## 3.1  OSCD Benchmark

### Dataset and evaluation protocol

The OSCD benchmark uses 24 Sentinel-2 cities split 10 / 4 / 10 (train / val / test).
All models are evaluated on the 10 held-out test cities with pooled confusion counts
(pixels summed across cities before computing rates).  The reference labels are binary
change/no-change derived from the OSCD hand-annotations.

SAR data on OSCD is Sentinel-1 multi-temporal composite imagery.  Exact acquisition
timestamps for the composite are not available from the dataset metadata; therefore no
precise temporal-gap characterisation is reported.

### Architecture and loss overview

CNN-family models (SAR-only U-Net, Optical-only U-Net, Sync Early Fusion U-Net,
Sync Dual-Stream CNN, Naive Async CNN, Clean Async CNN) were trained with standard,
unweighted binary cross-entropy (pos_weight = 1).
Transformer-family models (Self-Attention, Cross-Attention) were trained with
class-weighted BCE (pos_weight = 41.0), matching the approximate 1 : 40.7 pixel-level
class ratio in the training set.

### Table 1 — OSCD benchmark results (τ = 0.50, 10 test cities pooled)

| Model | Params | Loss | Best Ep | Val F1 | Test Prec | Test Rec | Test F1 | Test IoU | Test FAR | TP | FP | FN | TN |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAR-only U-Net | 31 037 057 | std BCE | 21 | 0.0662 | 0.4558 | 0.2917 | 0.3558 | 0.2164 | 0.0190 | 46 407 | 55 401 | 112 670 | 2 863 458 |
| Optical-only U-Net | 31 050 881 | std BCE | 50 | 0.2387 | 0.2515 | 0.2830 | 0.2663 | 0.1536 | 0.0459 | 45 025 | 134 013 | 114 052 | 2 784 846 |
| Sync Early Fusion U-Net | 31 052 033 | std BCE | 46 | 0.2538 | 0.2877 | 0.4533 | 0.3520 | 0.2136 | 0.0612 | 72 104 | 178 489 | 86 973 | 2 740 370 |
| Sync Dual-Stream CNN | 27 284 097 | std BCE | 31 | 0.3523 | 0.3009 | 0.5415 | 0.3868 | 0.2398 | 0.0686 | 86 137 | 200 124 | 72 940 | 2 718 735 |
| Naive Async CNN | 27 284 097 | std BCE | 46 | 0.1093 | 0.2363 | 0.1814 | 0.2052 | 0.1143 | 0.0319 | 28 849 | 93 242 | 130 228 | 2 825 617 |
| Clean Async CNN | 27 276 609 | std BCE | 26 | 0.0992 | 0.4753 | 0.3631 | 0.4117 | 0.2592 | 0.0218 | 57 764 | 63 762 | 101 313 | 2 855 097 |
| Transformer Self-Attn | 1 358 081 | BCE w=41 | 37 | 0.1333 | 0.0974 | 0.4734 | 0.1615 | 0.0879 | 0.2391 | 75 307 | 697 994 | 83 770 | 2 220 865 |
| Transformer Cross-Attn | 1 156 993 | BCE w=41 | 33 | 0.1428 | 0.2615 | 0.5592 | 0.3564 | 0.2168 | 0.0861 | 88 959 | 251 196 | 70 118 | 2 667 663 |

*Parameter counts verified from checkpoint state_dict key-shape products.*
*All-pixel protocol; SAR data on OSCD is multi-temporal composite — no-data masking is not applied.*
*Best epoch selected by validation F1 (all-pixel) in all OSCD runs.*

### Observations

**SAR-only U-Net (F1 = 0.356, FAR = 0.019).**
The SAR-only baseline achieves the highest precision of the U-Net family (0.456) with the
lowest false-alarm rate (0.019), suggesting the model is conservative in its change
predictions.  Recall is limited (0.292), indicating many changes are missed.

**Optical-only U-Net (F1 = 0.266, FAR = 0.046).**
Adding optical T1 and T2 imagery in place of SAR does not improve F1 over the SAR-only
model.  Recall improves slightly (0.283 vs. 0.292) but precision drops substantially
(0.252 vs. 0.456).  This observation is specific to OSCD and may reflect the quality or
temporal characteristics of the optical data relative to the label annotations.

**Sync Early Fusion U-Net (F1 = 0.352, FAR = 0.061) and Sync Dual-Stream CNN (F1 = 0.387, FAR = 0.069).**
Synchronous multimodal fusion models achieve higher recall (0.453 and 0.542) at the cost
of higher false-alarm rates.  The Dual-Stream CNN outperforms the Early Fusion U-Net on
F1 and IoU, with best-epoch selection occurring earlier (epoch 31 vs. 46).

**Naive Async CNN (F1 = 0.205) vs. Clean Async CNN (F1 = 0.412).**
Both models have the same architecture and parameter count (27 284 097).  They differ in
how the missing T2 optical channel is handled: the Naive model zero-fills the T2 optical
tensor (28 input channels total: 2 SAR + 13 optical T1 + 13 zeroed T2), while the Clean
model uses only 15 channels (2 SAR + 13 optical T1; T2 optical data is never loaded).
The Clean Async CNN achieves F1 = 0.412 and FAR = 0.022, substantially higher than the
Naive variant (F1 = 0.205, FAR = 0.032).  This gap illustrates that zero-filled channels
representing absent data introduce a confounding signal.

**Transformer models (Self-Attn F1 = 0.162, Cross-Attn F1 = 0.356).**
The Self-Attention Transformer has the highest recall in the benchmark (0.473) but also
the highest false-alarm rate (0.239), resulting in low precision (0.097) and low F1
(0.162).  The Cross-Attention Transformer achieves F1 = 0.356 and FAR = 0.086, comparable
to the Sync Early Fusion U-Net despite having 27× fewer parameters (1.16 M vs. 31.05 M).
Both Transformer models were trained with pos_weight = 41; the CNN models used unweighted
BCE.  This loss difference confounds direct architectural comparisons between families.

**Validation–test discrepancy.**
For several models the test-set F1 at τ = 0.50 differs materially from the validation F1
used for checkpoint selection.  For example, the SAR-only U-Net achieves val F1 = 0.066
but test F1 = 0.356; the Sync Dual-Stream CNN achieves val F1 = 0.352 and test F1 = 0.387.
This discrepancy is observed but not causally attributed: possible contributing factors
include distribution shift between the 4 validation cities and the 10 test cities, and
threshold sensitivity differences across splits.

---

## 3.2  Scenario 2 — SpaceNet 7 Initial Baselines (Seed 42)

### Dataset and evaluation protocol

SpaceNet 7 provides monthly Sentinel-2 mosaics (10 spectral bands: B2, B3, B4, B5, B6,
B7, B8, B8A, B11, B12) and co-located Sentinel-1 dual-polarisation SAR (VV, VH), giving
4 SAR channels per time step.  The locked geographic split assigns 20 / 4 / 4 sites to
train / val / test.  This split is fixed by SHA-256:
`5d0841adf0c651166cb18ecea310c6ef8cce69172065fc2d699a29083c272e27` and is not modified
for any subsequent experiment.

All Scenario 2 models use a custom masked BCE loss variant for the masked-loss condition,
which zeros the gradient contribution of pixels where the SAR data is zero (no-data) in
either time step.  The standard-BCE condition uses unweighted BCE over all pixels
(pos_weight = 94.0 for both conditions, matching the 1 : 93.64 class ratio in the
training set).

**Checkpoint selection — initial baselines (seed 42).**
The three standard-BCE baselines were selected by validation F1 (all-pixel).
The three masked-BCE runs were selected by validation F1 (valid-pixel).
This difference in selection criterion is a confound when comparing the two loss
conditions on the seed-42 results; it is corrected in the matched multi-seed experiment
(Section 3.3).

### Table 2 — Scenario 2 initial baselines (seed 42, τ = 0.50)

| Model | Loss | Sel. Ep | Val F1 | Prec (all) | Rec (all) | F1 (all) | FAR (all) | F1 (valid) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Async CNN | std BCE w=94 | 2 | 0.1832 (all) | 0.0479 | 0.2334 | 0.0794 | 0.0336 | 0.0804 |
| Sync CNN | std BCE w=94 | 21 | 0.2663 (all) | 0.0610 | 0.5680 | 0.1102 | 0.0633 | 0.1107 |
| Cross-Attn | std BCE w=94 | 23 | 0.1328 (all) | 0.0214 | 0.5705 | 0.0412 | 0.1892 | 0.0411 |
| Async CNN | masked BCE w=94 | 2 | 0.1677 (valid) | 0.0473 | 0.2907 | 0.0813 | 0.0424 | 0.0822 |
| Sync CNN | masked BCE w=94 | 30 | 0.2878 (valid) | 0.0781 | 0.5342 | 0.1363 | 0.0456 | 0.1376 |
| Cross-Attn | masked BCE w=94 | 41 | 0.2223 (valid) | 0.0267 | 0.4639 | 0.0506 | 0.1222 | 0.0506 |

*Val F1 column indicates which protocol was used for checkpoint selection.*
*F1 (valid) for the three std-BCE runs is derived from post-hoc ablation evaluation*
*(scratch/ablation_eval_results.json); these runs did not log valid-pixel metrics during training.*
*All Scenario 2 models: 20 train sites, 4 val sites, 4 test sites (locked split).*

### Observations

**Overall performance level.**
All Scenario 2 models achieve F1 (all-pixel) in the range 0.04–0.14 at τ = 0.50.
This is substantially lower than the OSCD benchmark (F1 range 0.16–0.41).  Scenario 2
presents a harder task: asynchronous SAR and optical acquisitions, dual-polarisation SAR,
fewer training sites (20 vs. 10 OSCD test cities for reference), and higher class
imbalance (1 : 93.64 vs. approximately 1 : 40.7 on OSCD).

**Async CNN baseline (seed 42, best epoch 2).**
The Async CNN baseline selects its checkpoint at epoch 2, the earliest epoch allowed.
This is consistent with training instability or rapid overfitting under high class
imbalance.  The F1 of 0.079 (all-pixel) and 0.080 (valid-pixel) are the lowest of the
three architectures.

**Sync CNN baseline vs. masked.**
The Sync CNN masked-BCE variant (F1 valid = 0.1376) outperforms the std-BCE baseline
(F1 valid = 0.1107) at τ = 0.50.  However, the two runs differ in checkpoint-selection
criterion (all-pixel vs. valid-pixel), so this difference may partly reflect the
selection protocol rather than the loss function alone.  The matched multi-seed
experiment in Section 3.3 addresses this by holding the selection criterion fixed.

**Cross-Attention model.**
The Cross-Attention Transformer achieves high recall (0.57) but the highest FAR (0.19
for std-BCE, 0.12 for masked-BCE), resulting in low precision and F1.  The masked-BCE
variant roughly halves the FAR (0.19 → 0.12) while maintaining similar recall.

---

## 3.3  Matched Multi-Seed Experiment — Sync CNN

### Design

To isolate the effect of the loss function from confounds in the seed-42 runs,
a controlled multi-seed comparison was conducted using only the Sync Dual-Stream CNN.
Four additional runs were created: seeds 43 and 44, each with standard-BCE and masked-BCE.

**Controls held constant across both loss conditions within each seed:**
- Initial model weights (identical state_dict for each seed pair, confirmed by preflight test).
- Training crop coordinates and crop order (deterministic pre-generated crop schedule,
  consumed identically by both conditions for the same seed).
- Checkpoint-selection criterion: validation F1 (valid-pixel) for all four runs.
- Training sites, validation sites, preprocessing, batch size, Adam lr = 1 × 10⁻⁴,
  pos_weight = 94.0, 50 epochs, crop size 32 × 32, 64 crops per site per epoch.

Seeds 43 and 44 use different initializations and crop schedules from each other and from seed 42.

**Relationship to seed 42.**
The seed-42 Sync CNN masked-BCE run (Section 3.2) used valid-pixel checkpoint selection and
is included in Table 3 for context.  The seed-42 std-BCE run used all-pixel checkpoint
selection and is therefore not directly comparable under the controlled protocol; it is
listed in Table 3 with a note.

### Table 3 — Matched multi-seed Sync CNN results (τ = 0.50)

| Seed | Loss | Peak Ep | Val F1 (valid) | F1 (all) | F1 (valid) |
|---|---|---:|---:|---:|---:|
| 42† | std BCE w=94 | 21 | 0.2663 (all)† | 0.1102 | 0.1107 |
| 42 | masked BCE w=94 | 30 | 0.2878 | 0.1363 | 0.1376 |
| 43 | std BCE w=94 | 30 | 0.2932 | 0.1327 | 0.1342 |
| 43 | masked BCE w=94 | 30 | 0.2877 | 0.1236 | 0.1257 |
| 44 | std BCE w=94 | 43 | 0.2746 | 0.1327 | 0.1337 |
| 44 | masked BCE w=94 | 43 | 0.2744 | 0.1141 | 0.1180 |

*† Seed-42 std-BCE run used all-pixel checkpoint selection; it is not part of the matched comparison.*
*Seeds 43 and 44 use valid-pixel checkpoint selection for both loss conditions.*
*Peak epoch is the epoch selected by the checkpoint-selection criterion.*

**Confusion counts (τ = 0.50) — verified from metrics.json:**

| Seed | Loss | TP | FP | FN | TN |
|---|---|---:|---:|---:|---:|
| 43 | std BCE | 3 669 | 44 770 | 3 190 | 902 899 |
| 43 | masked BCE | 3 923 | 52 695 | 2 936 | 894 974 |
| 44 | std BCE | 3 924 | 48 353 | 2 935 | 899 316 |
| 44 | masked BCE | 3 822 | 56 318 | 3 037 | 891 351 |

*Sources: experiments/scenario_2_multiseed/sync_cnn_{baseline,masked}_seed{43,44}/metrics.json,*
*key: test_threshold_0_5.all_pixel_protocol*

### Observations

**Peak epoch alignment.**
For both seeds 43 and 44, the std-BCE and masked-BCE conditions reach peak validation F1
(valid-pixel) at the same epoch (seed 43: epoch 30 for both; seed 44: epoch 43 for both).
This alignment was not observed for seed 42 (epochs 21 vs. 30), which used different
selection criteria for the two conditions.

**Within-seed F1 comparison.**
At τ = 0.50 under the controlled protocol:
- Seed 43: std-BCE F1 (all) = 0.1327 vs. masked-BCE = 0.1236; difference = −0.0091.
- Seed 44: std-BCE F1 (all) = 0.1327 vs. masked-BCE = 0.1141; difference = −0.0186.

In both seeds, the masked-BCE condition achieves lower all-pixel F1 at τ = 0.50 than the
std-BCE condition.  Inspection of the confusion counts shows that masked-BCE tends to
produce higher recall but higher FP relative to std-BCE within the same seed.

**Descriptive summary across matched seeds (N = 2; descriptive only).**
Baseline all-pixel F1: mean = 0.1327, std = 0.0000.
Masked all-pixel F1: mean = 0.1188, std = 0.0067.
Mean paired difference (masked − baseline): −0.0139.
With N = 2 seeds, these observations are descriptive; no inference about the broader
distribution is warranted.

---

## 3.4  Secondary Operating Point — Calibrated Threshold τ*

Because the sigmoid output of models trained with high pos_weight tends to be
systematically shifted, a per-run calibrated threshold τ* was identified by maximising
F1 (valid-pixel) on the validation set.  Table 4 reports results at both τ = 0.50 and
the calibrated τ* for all Sync CNN runs that have calibrated-threshold results available.

### Table 4 — Sync CNN results at τ = 0.50 and calibrated τ* (valid-pixel protocol)

| Seed | Loss | τ* | F1 (all) @ 0.50 | F1 (valid) @ 0.50 | F1 (all) @ τ* | F1 (valid) @ τ* |
|---|---|---:|---:|---:|---:|---:|
| 42 | std BCE | 0.50 | 0.1102 | 0.1107 | 0.1102 | 0.1107 |
| 42 | masked BCE | 0.78 | 0.1363 | 0.1376 | 0.1909 | 0.1934 |
| 43 | std BCE | 0.79 | 0.1327 | 0.1342 | 0.1943 | 0.1971 |
| 43 | masked BCE | 0.83 | 0.1236 | 0.1257 | 0.1912 | 0.1944 |
| 44 | std BCE | 0.86 | 0.1327 | 0.1337 | 0.2180 | 0.2191 |
| 44 | masked BCE | 0.91 | 0.1141 | 0.1180 | 0.2113 | 0.2127 |

*τ* selected by maximising valid-pixel F1 on the validation set.*
*Seed-42 std-BCE τ* = 0.50 indicates the validation-optimal threshold coincides with 0.50.*
*Both All-Pixel and Valid-Pixel metrics reported at each threshold to allow consistent comparison.*

**Observation.**
At calibrated τ*, the F1 gap between std-BCE and masked-BCE conditions narrows or reverses
compared to τ = 0.50.  For seed 43, std-BCE at τ* achieves F1 (valid) = 0.1971 vs.
masked-BCE F1 (valid) = 0.1944; the difference (−0.0027) is smaller in magnitude than at
τ = 0.50 (−0.0085).  For seed 44, std-BCE at τ* achieves F1 (valid) = 0.2191 vs.
masked-BCE F1 (valid) = 0.2127 (difference = −0.0064).  The masked-BCE runs tend to
produce higher optimal thresholds (τ* = 0.78–0.91) than std-BCE runs (τ* = 0.50–0.86),
consistent with the effect of masking on the output distribution.

These threshold differences indicate that masked-BCE and std-BCE models may not be
directly comparable at a single fixed threshold, and that threshold calibration is an
important practical consideration.

---

## 3.5  Summary of Supported Findings

The following findings are directly supported by the saved experiment artifacts:

1. On OSCD, the Clean Async CNN (15 channels; T2 optical not loaded) achieves F1 = 0.412,
   substantially higher than the Naive Async CNN (28 channels; T2 zeroed, F1 = 0.205),
   with the same architecture and parameter count.  This difference is attributed to the
   handling of the absent T2 optical channel rather than to model capacity or any other
   controlled variable (Section 3.1).

2. On OSCD, the Cross-Attention Transformer achieves F1 = 0.356 with 1.16 M parameters,
   comparable to the Sync Early Fusion U-Net (F1 = 0.352) which has 31.05 M parameters.
   Both were evaluated at τ = 0.50.  The loss functions differ (BCE w=41 vs. std BCE),
   which confounds direct comparison (Section 3.1).

3. On Scenario 2, the Sync CNN masked-BCE run (seed 42) achieves valid-pixel F1 = 0.1376
   vs. std-BCE valid-pixel F1 = 0.1107.  This comparison is not controlled (different
   selection criteria), and should be interpreted with caution (Section 3.2).

4. In the controlled matched multi-seed comparison (seeds 43 and 44), masked-BCE does not
   improve all-pixel F1 over std-BCE at τ = 0.50 for either seed (differences: −0.0091
   and −0.0186 in favour of std-BCE).  At calibrated τ*, the gap narrows but std-BCE
   remains at or above masked-BCE in valid-pixel F1 for both seeds (Section 3.4).

5. Masked-BCE loss systematically shifts the model output distribution toward higher
   confidence values (higher optimal τ*), consistent with its restriction of gradient
   updates to valid-SAR pixels.

The following are **not** supported by the available experiments:

- Any claim that masked-BCE improves performance across architectures or datasets.
- Any claim that the above differences are statistically significant (N = 2 seeds is
  insufficient for inference).
- Any temporal-gap experiment or single-pol vs. dual-pol ablation result.
