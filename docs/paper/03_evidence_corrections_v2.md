# Evidence Corrections v2

This document extends `03_evidence_corrections.md` (v1) with the additional corrections
identified during the scientific consistency audit of October 2026. Items C1–C7 from v1
are carried forward with minor clarifications. Items C8–C14 are new.

For each item:
- **Original** — exact statement in v1 docs or PAPER_DRAFTING_FOUNDATION.md
- **Corrected** — verified replacement statement
- **Evidence** — file(s) and line/key confirming the correction
- **Verified?** — directly verified from saved artifacts vs. uncertain/inferred

---

## C1 — OSCD Loss Attribution (carried from v1, clarified)

**Original:** All OSCD models were trained with class-weighted BCE with pos_weight = 41.

**Corrected:** CNN-family OSCD models (SAR-only U-Net, Optical-only U-Net, Sync Early
Fusion U-Net, Sync Dual-Stream CNN, Naive Async CNN, Clean Async CNN) were trained with
standard unweighted BCE (pos_weight = 1; `pos_weight` key absent from all CNN config
files). Both Transformer models (Self-Attention, Cross-Attention) used weighted BCE with
pos_weight = 41.0.

**Evidence:**
- All six CNN `config.yaml` files: no `pos_weight` key (confirmed: sar_baseline,
  optical_baseline, fusion_baseline, dual_stream_baseline, asynchronous_dual_stream,
  asynchronous_dual_stream_clean).
- `experiments/transformer_self_attention/config.yaml`: `pos_weight: 41.0`
- `experiments/transformer_cross_attention/config.yaml`: `pos_weight: 41.0`

**Verified?** Directly verified.

---

## C2 — OSCD City Count and Split (carried from v1)

**Original:** OSCD uses a 14 / 10 train/test split.

**Corrected:** OSCD uses a 10 / 4 / 10 train / validation / test split (24 cities total).

**Evidence:** `experiments/sar_baseline/config.yaml` — `train_cities` (10 entries),
`val_cities` (4 entries), `test_cities` (10 entries). Consistent across all OSCD configs.

**Verified?** Directly verified.

---

## C3 — SAR-only U-Net Parameter Count (carried from v1)

**Original:** 31,036,545 parameters.
**Corrected:** 31,037,057 parameters (difference: +512).

**Evidence:** Checkpoint state_dict from `experiments/sar_baseline/checkpoints/best_model.pth`;
cross-checked against `results/tables/experiment_summary.json`.

**Verified?** Directly verified.

---

## C4 — Optical-only U-Net Parameter Count (carried from v1)

**Original:** 31,051,969 parameters.
**Corrected:** 31,050,881 parameters (difference: −1,088).

**Evidence:** Checkpoint state_dict from `experiments/optical_baseline/checkpoints/best_model.pth`;
cross-checked against `results/tables/experiment_summary.json`.

**Verified?** Directly verified.

---

## C5 — Scenario 2 Parameter Counts (carried from v1)

**Original:** Scenario 2 counts listed identically to Scenario 1 (27,284,097 for both CNN
variants; 1,156,993 for Cross-Attention).

**Corrected:**

| Model | Scenario 1 (OSCD) | Scenario 2 (SpaceNet 7) |
|---|---:|---:|
| Async Dual-Stream CNN | 27,284,097 | 27,276,033 |
| Sync Dual-Stream CNN | 27,284,097 | 27,281,793 |
| Cross-Attention Transformer | 1,156,993 | 1,154,945 |

The difference arises from the different input channel counts (4-channel dual-pol SAR and
10-band optical in Scenario 2 vs. 2-channel SAR and 13-band optical in OSCD).

**Evidence:** Checkpoint state_dicts from `experiments/scenario_2/{async_dual_stream,
sync_dual_stream,cross_attention}/checkpoints/best_model.pth`.

**Verified?** Directly verified.

---

## C6 — Scenario 2 Sync CNN Baseline Valid-Pixel F1 (carried from v1)

**Original:** 0.1102 (all-pixel value copied into valid-pixel column).
**Corrected:** 0.1107.

**Evidence:** `scratch/ablation_eval_results.json`,
key `sync_cnn_baseline_seed42.valid_pixel_protocol.f1`.

**Verified?** Directly verified.

---

## C7 — Calibrated τ* Table Missing Valid-Pixel Column (carried from v1)

**Original:** Table reported all-pixel F1 only at τ*.
**Corrected:** Both all-pixel and valid-pixel F1 reported at τ* (Table 4 in 02_results_v2.md).

**Evidence:** Both protocols available in `experiments/scenario_2_multiseed/*/metrics.json`.

**Verified?** Directly verified.

---

## C8 — Naive vs. Clean Async CNN Parameter Count Claim (NEW)

**Original statement (02_results.md v1, §3.1, Naive vs. Clean paragraph):**
"Both models have the same architecture and parameter count (27,284,097)."

**Corrected statement:**
The two models have the same architectural topology (Dual-Stream U-Net with identical
encoder and decoder stage dimensions) but different parameter counts: 27,284,097 (Naive,
28-channel input) vs. 27,276,609 (Clean, 15-channel input). The parameter difference of
7,488 arises from the different input width of the optical encoder's first convolution
layer.

**Evidence:**
- `results/tables/experiment_summary.json`: `num_parameters` field for both models.
- `src/models/dual_stream_unet_clean.py`, docstring line 23: "27,276,609 (exact difference
  of 7,488 weights from 13 unused input channels)."

**Verified?** Directly verified.

**Impact:** The Naive vs. Clean comparison is between two different input configurations
with different first-layer parameter counts. The F1 difference cannot be attributed to
zero-filling alone. The v2 results section acknowledges both differences.

---

## C9 — Naive vs. Clean Comparison Described as Isolating Zero-Filling (NEW)

**Original statement (02_results.md v1, §3.1):**
"This gap illustrates that zero-filled channels representing absent data introduce a
confounding signal."

**Original statement (02_results.md v1, §3.5, item 1):**
"This difference is attributed to the handling of the absent T2 optical channel rather
than to model capacity or any other controlled variable."

**Corrected statement:**
The observed F1 difference between the Naive and Clean Async CNN is associated with
both the zero-filled input channels and the resulting first-layer parameter count
difference. The experiment does not hold these two factors constant separately; it is not
a controlled ablation of zero-filling alone. The comparison is between two input
configurations.

**Evidence:** See C8. No single-factor ablation was run (e.g., no model that uses 28
input channels but initialises the 13 T2 optical channels to a learned constant, or a
model that uses 13 T2 optical channels but with the same input width as the naive).

**Verified?** Uncertainty confirmed from design; no ablation artifact exists.

---

## C10 — Scenario 2 Standard Loss Described as Unweighted (NEW)

**Original statements (02_results.md v1, §3.2 dataset protocol paragraph):**
"The standard-BCE condition uses unweighted BCE over all pixels (pos_weight = 94.0 for
both conditions…)"

*[Note: this was internally contradictory — it says "unweighted" then lists pos_weight=94.
The v1 phrasing was inconsistent.]*

**Corrected statement:**
Both the standard and masked loss conditions use weighted BCE with pos_weight = 94.0.
The standard condition uses `nn.BCEWithLogitsLoss(pos_weight=94.0)` (all pixels
contribute to numerator and denominator). The masked condition uses
`MaskedBCEWithLogitsLoss(pos_weight=94.0)` (only valid-SAR pixels contribute to
numerator; denominator is valid pixel count).

**Evidence:**
- `src/training/train_scenario2_dual_stream.py`, line 344–345:
  `pos_weight = torch.tensor([POS_WEIGHT]).to(device)` and
  `criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)`.
- `src/training/train_scenario2_sync_multiseed.py`, lines 317–322:
  Both branches set `pos_weight = torch.tensor([POS_WEIGHT]).to(device)`;
  masked branch uses `MaskedBCEWithLogitsLoss(pos_weight=pos_weight)`,
  standard branch uses `nn.BCEWithLogitsLoss(pos_weight=pos_weight)`.
- `POS_WEIGHT = 94.0` in both training scripts.

**Verified?** Directly verified from source code.

---

## C11 — Optical-only Recall Comparison Stated Backwards (NEW)

**Original statement (02_results.md v1, §3.1, Optical-only paragraph):**
"Recall improves slightly (0.283 vs. 0.292)…"

**Corrected statement:**
Recall decreases: Optical-only recall = 0.2830, SAR-only recall = 0.2917.
0.283 < 0.292; optical-only recall is lower, not higher.

**Evidence:** Table 1 confusion counts, both in v1 and v2:
- Optical-only U-Net: TP=45,025, FN=114,052 → Recall = 45,025 / (45,025+114,052) = 0.2830
- SAR-only U-Net: TP=46,407, FN=112,670 → Recall = 46,407 / (46,407+112,670) = 0.2917

**Verified?** Directly verified by arithmetic from saved confusion counts.

---

## C12 — Valid-Pixel Definition Based on Zero-SAR Rather Than NaN (NEW)

**Original statement (02_results.md v1, §Notation):**
"Valid-pixel — denominator restricted to pixels where SAR data is non-zero in *both*
time steps; pixels with zeroed SAR (no-data) are excluded…"

**Corrected statement:**
Validity is determined by the original NaN indicators in the raw raster, recorded before
NaN-to-zero replacement. Specifically, a pixel is valid if neither the VV nor the VH
channel is NaN at either time step in the original raster. A SAR pixel that is genuinely
zero-valued (i.e., not NaN in the original raster) is treated as valid, not excluded.
After validity is recorded, NaNs are replaced with 0.0 in the input tensor.

**Evidence:**
`src/datasets/scenario2_dataset.py`, function `_read_s1_with_validity`, line 113:
```python
valid_mask = ~np.any(np.isnan(arr), axis=0)  # (H, W)
arr = np.nan_to_num(arr, nan=0.0)
```
The validity mask is constructed from `np.isnan` checks *before* `nan_to_num` is called.

**Verified?** Directly verified from source code.

**Impact:** The v1 definition ("non-zero in both time steps") would incorrectly exclude
valid SAR pixels whose backscatter is genuinely at or near zero. The v2 definition
correctly reflects the implementation.

---

## C13 — Cross-Dataset Difficulty Claim (NEW)

**Original statement (02_results.md v1, §3.2 Observations):**
"Scenario 2 presents a harder task: asynchronous SAR and optical acquisitions,
dual-polarisation SAR, fewer training sites (20 vs. 10 OSCD test cities for reference),
and higher class imbalance…"

**Corrected statement:**
OSCD and Scenario 2 differ in label type, sensor configuration, geographic scope, and
class imbalance. Their F1 scores are not directly comparable because the benchmarks are
not controlled for any single factor. The lower absolute F1 on Scenario 2 is consistent
with these differences but does not, on its own, establish that Scenario 2 is inherently
harder. The comparison of training-site counts (20 Scenario 2 sites vs. 10 OSCD test
cities) is also not meaningful because these are train-set and test-set sizes from
different experiments.

**Evidence:** No cross-dataset controlled experiment was conducted. The characterisation
as "harder" is an unsupported inference.

**Verified?** Uncertainty confirmed — no ablation exists.

---

## C14 — Threshold Calibration Attributed to a Mechanism (NEW)

**Original statement (02_results.md v1, §3.4):**
"The masked-BCE runs tend to produce higher optimal thresholds (τ* = 0.78–0.91) than
std-BCE runs (τ* = 0.50–0.86), consistent with the effect of masking on the output
distribution."

**Corrected statement:**
The masked-loss runs in the matched experiment select higher calibrated thresholds than
the corresponding std-loss runs (seed 43: 0.83 vs. 0.79; seed 44: 0.91 vs. 0.86). This
is an observation across two seeds. The phrase "consistent with the effect of masking on
the output distribution" implies a mechanistic explanation that is not supported by a
direct analysis of the output distribution (e.g., probability calibration curves, score
histograms). The statement should be recorded as an observation only.

**Evidence:** Calibrated threshold values in `experiments/scenario_2_multiseed/*/metrics.json`.
No output distribution analysis artifact exists.

**Verified?** Observation verified; mechanism not established.

---

## Summary of All Corrections (v2)

| ID | File affected | Original claim | Corrected claim | Directly verified? |
|---|---|---|---|---|
| C1 | 01_methods | All OSCD models used pos_weight=41 | CNN: w=1; Transformer: w=41 | Yes |
| C2 | 01_methods | 14/10 OSCD split | 10/4/10, 24 cities | Yes |
| C3 | 01_methods | SAR-only params: 31,036,545 | 31,037,057 | Yes |
| C4 | 01_methods | Optical-only params: 31,051,969 | 31,050,881 | Yes |
| C5 | 01_methods | S2 params copied from S1 | S2-specific counts (see table) | Yes |
| C6 | 02_results | S2 Sync CNN valid-pixel F1: 0.1102 | 0.1107 | Yes |
| C7 | 02_results | τ* table all-pixel only | Both protocols in Table 4 | Yes |
| C8 | 02_results | Naive & Clean CNNs: same param count | Different by 7,488 params | Yes |
| C9 | 02_results | F1 gap isolates zero-filling | Combined effect of input config + params | Confirmed uncertain |
| C10 | 01_methods, 02_results | S2 standard condition is "unweighted BCE" | Both conditions use pos_weight=94.0 | Yes |
| C11 | 02_results | Optical-only recall improves over SAR-only | Optical-only recall is lower (0.283 < 0.292) | Yes |
| C12 | 01_methods, 02_results | Valid-pixel based on non-zero SAR | Based on pre-fill NaN indicators | Yes |
| C13 | 02_results | Scenario 2 is a "harder task" | Datasets are not comparable; no controlled comparison | Confirmed uncertain |
| C14 | 02_results | τ* pattern attributed to masking mechanism | Observation only; mechanism not established | Observation verified |

---

## Items Checked and Confirmed Correct (v2, no change)

- pos_weight = 94.0 for all Scenario 2 loss conditions.
- Locked split SHA-256: 5d0841adf0c651166cb18ecea310c6ef8cce69172065fc2d699a29083c272e27.
- Adam lr=1e-4, batch_size=64, crop_size=32, 50 epochs — all Scenario 2 configs.
- 64 crops per site per epoch.
- Scenario 2 test confusion counts for seeds 43 and 44 (recomputed F1 matches stored to <1e-7).
- OSCD class ratio ≈ 1:40.7 → pos_weight=41.
- S2 class ratio 93.64 → pos_weight=94.
- Scenario 2 SAR: dual-pol VV+VH = 4 SAR channels per time step.
- Scenario 2 optical: 10-band subset (B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12).
- Checkpoint selection for seeds 43, 44: val_f1_valid for all four matched runs.
- Scenario 2 label derivation: XOR of thresholded building probability maps (threshold 0.5).
- T1 = March 2018, T2 = December 2019, 21-month separation, uniform across 28 sites.
- OSCD SAR is single-pol (VV only) composite; no exact timestamps available.
- RQ2 (temporal gap) and RQ3 (polarisation) are untested by the available experiments.
