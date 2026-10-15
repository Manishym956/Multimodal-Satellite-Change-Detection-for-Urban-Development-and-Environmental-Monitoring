# Evidence Corrections

This document records every discrepancy found between the claims in
`docs/PAPER_DRAFTING_FOUNDATION.md` and the values verified from saved experiment
artifacts.  For each discrepancy the original claim, the corrected value, the evidence
source, and the verification status are listed.

All corrections are incorporated into `docs/paper/01_methods.md` and
`docs/paper/02_results.md`.  The foundation document is **not** modified.

---

## Correction 1 — OSCD Loss Attribution

**Original claim (Foundation):**
All OSCD models were trained with class-weighted BCE with pos_weight = 41.

**Corrected statement:**
CNN-family models (SAR-only U-Net, Optical-only U-Net, Sync Early Fusion U-Net,
Sync Dual-Stream CNN, Naive Async CNN, Clean Async CNN) were trained with standard,
unweighted binary cross-entropy (pos_weight absent from config).  Only the two
Transformer models (Self-Attention, Cross-Attention) used BCE with pos_weight = 41.0.

**Evidence:**
- `experiments/sar_baseline/config.yaml` — no pos_weight key.
- `experiments/optical_baseline/config.yaml` — no pos_weight key.
- `experiments/fusion_baseline/config.yaml` — no pos_weight key.
- `experiments/dual_stream_baseline/config.yaml` — no pos_weight key.
- `experiments/asynchronous_dual_stream/config.yaml` — no pos_weight key.
- `experiments/asynchronous_dual_stream_clean/config.yaml` — no pos_weight key.
- `experiments/transformer_self_attention/config.yaml` — pos_weight: 41.0 present.
- `experiments/transformer_cross_attention/config.yaml` — pos_weight: 41.0 present.

**Verification status:** Directly verified from config files.

**Impact:** The OSCD result table (Table 1 in 02_results.md) lists loss settings per
model.  Any cross-family F1 comparison (CNN vs. Transformer) is confounded by the
differing loss functions and should not be interpreted as a purely architectural effect.

---

## Correction 2 — OSCD City Count and Split

**Original claim (Foundation):**
OSCD uses a 14 / 10 train/test split.

**Corrected statement:**
OSCD uses a 10 / 4 / 10 train / validation / test split over 24 cities.
The 4 validation cities are used for checkpoint selection only; they are not test cities.

**Evidence:**
- `experiments/sar_baseline/config.yaml`: train_cities (10 entries), val_cities (4 entries), test_cities (10 entries).
- Cross-checked against OSCD dataset README and consistent across all OSCD experiment configs.

**Verification status:** Directly verified from config files.

**Impact:** Methods section must state 24 total cities, 10 / 4 / 10 split.  The
Foundation document omitted the validation split and under-counted test cities.

---

## Correction 3 — SAR-only U-Net Parameter Count

**Original claim (Foundation):**
SAR-only U-Net: 31 036 545 parameters.

**Corrected value:**
31 037 057 parameters (difference: +512).

**Evidence:**
Checkpoint state_dict loaded from
`experiments/sar_baseline/checkpoints/best_model.pth`; parameter count computed as the
sum of the product of each tensor's shape elements over all keys in the state_dict.
Cross-checked against `results/tables/experiment_summary.json` (field: `num_parameters`).

**Verification status:** Directly verified from checkpoint.

**Impact:** Methods table must use 31 037 057.

---

## Correction 4 — Optical-only U-Net Parameter Count

**Original claim (Foundation):**
Optical-only U-Net: 31 051 969 parameters.

**Corrected value:**
31 050 881 parameters (difference: −1 088).

**Evidence:**
Checkpoint state_dict loaded from
`experiments/optical_baseline/checkpoints/best_model.pth`; parameter count computed
identically to Correction 3.
Cross-checked against `results/tables/experiment_summary.json`.

**Verification status:** Directly verified from checkpoint.

**Impact:** Methods table must use 31 050 881.

---

## Correction 5 — Scenario 2 Parameter Counts (Copied from Scenario 1)

**Original claim (Foundation):**
Scenario 2 model parameter counts were listed as identical to Scenario 1 values.
Specifically, the Foundation stated:
- Async Dual-Stream CNN: 27 284 097
- Sync Dual-Stream CNN: 27 284 097

**Corrected values:**
Scenario 2 architectures differ from Scenario 1 in input channel count (4 SAR channels
for dual-pol VV+VH, vs. 2 SAR channels for VV-only on OSCD) and in the number of
optical bands (10 vs. 13), altering the first-layer weights.  The verified counts are:

| Model | Scenario 1 (OSCD) | Scenario 2 (SpaceNet 7) |
|---|---:|---:|
| Async Dual-Stream CNN | 27 284 097 | 27 276 033 |
| Sync Dual-Stream CNN | 27 284 097 | 27 281 793 |
| Cross-Attention Transformer | 1 156 993 | 1 154 945 |

**Evidence:**
Checkpoint state_dict loaded from:
- `experiments/scenario_2/async_dual_stream/checkpoints/best_model.pth` → 27 276 033
- `experiments/scenario_2/sync_dual_stream/checkpoints/best_model.pth` → 27 281 793
- `experiments/scenario_2/cross_attention/checkpoints/best_model.pth` → 1 154 945
Cross-checked against `experiments/scenario_2/*/config.yaml` (in_channels fields).

**Verification status:** Directly verified from checkpoints.

**Impact:** Scenario 2 Methods table must use the corrected values above.
Scenario 1 (OSCD) values are unaffected.

---

## Correction 6 — Scenario 2 Sync CNN Baseline Valid-Pixel F1

**Original claim (Foundation):**
Sync CNN baseline (seed 42, std-BCE) test F1 (valid-pixel) = 0.1102.

**Corrected value:**
0.1107.

The Foundation appears to have copied the all-pixel F1 (0.1102) into the valid-pixel
column.

**Evidence:**
Post-hoc ablation evaluation results in
`scratch/ablation_eval_results.json`, key path:
`sync_cnn_baseline_seed42.valid_pixel_protocol.f1` = 0.1107.
Confusion counts from `experiments/scenario_2/sync_dual_stream/metrics.json` were
used to recompute both protocols; the all-pixel F1 is 0.1102 and the valid-pixel F1
is 0.1107 (they differ because the valid-pixel denominator excludes no-data pixels).

**Verification status:** Directly verified from saved ablation results and confusion counts.

**Impact:** Table 2 in 02_results.md uses 0.1107 for the valid-pixel column.

---

## Correction 7 — Calibrated τ* Table Missing Valid-Pixel Column

**Original claim (Foundation):**
The calibrated-threshold table reported only all-pixel F1 at τ*.

**Corrected statement:**
To allow consistent comparison between runs and with the primary τ = 0.50 results,
both all-pixel and valid-pixel F1 must be reported at each threshold.  This is
especially important because the checkpoint-selection criterion for the matched runs
is valid-pixel F1.

**Evidence:**
Both all-pixel and valid-pixel confusion counts are available at each threshold in
`experiments/scenario_2_multiseed/*/metrics.json`.

**Verification status:** Corrected by explicit dual-column reporting in Table 4 of
02_results.md.

**Impact:** Table 4 in 02_results.md includes both F1 (all) @ τ* and F1 (valid) @ τ*
for every run.

---

## Summary Table

| # | Location in Foundation | Original | Corrected | Evidence File | Directly Verified? |
|---|---|---|---|---|---|
| 1 | Loss settings (OSCD) | All models: pos_weight=41 | CNN: std BCE; Transformer: w=41 | experiments/*/config.yaml | Yes |
| 2 | OSCD split | 14 / 10 | 10 / 4 / 10 (24 cities) | experiments/sar_baseline/config.yaml | Yes |
| 3 | SAR-only U-Net params | 31 036 545 | 31 037 057 | checkpoint state_dict | Yes |
| 4 | Optical-only U-Net params | 31 051 969 | 31 050 881 | checkpoint state_dict | Yes |
| 5 | Scenario 2 params | Copied from S1 | See table above | S2 checkpoint state_dicts | Yes |
| 6 | S2 Sync CNN Baseline F1 (valid) | 0.1102 | 0.1107 | ablation_eval_results.json | Yes |
| 7 | Calibrated τ* table columns | All-pixel only | Both all-pixel and valid-pixel | multiseed metrics.json | Yes |

---

## Notes on Non-Corrections

The following items in the Foundation were checked and found to be consistent with the
saved artifacts:

- **pos_weight = 94.0 for Scenario 2:** Verified from all Scenario 2 config.yaml files.
- **Locked split SHA-256:** 5d0841adf0c651166cb18ecea310c6ef8cce69172065fc2d699a29083c272e27 verified from split_definition.json.
- **Adam lr = 1e-4, batch_size = 64, crop_size = 32, 50 epochs:** Verified from config.yaml for all runs.
- **64 crops per site per epoch:** Verified from config.yaml (crops_per_site: 64).
- **Scenario 2 test confusion counts for seeds 43 and 44:** Verified by recomputation.
- **OSCD class ratio ≈ 1 : 40.7 (→ pos_weight = 41):** Consistent with dataset statistics.
- **S2 class ratio 93.64 (→ pos_weight = 94):** Verified from split_definition.json class counts (50 387 change / 4 718 348 no-change).
- **Scenario 2 SAR: dual-pol (VV + VH) = 4 SAR channels per time step:** Verified from config.yaml in_channels.
- **Scenario 2 optical: 10-band (B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12):** Verified from config.yaml band_indices.
- **Checkpoint selection criterion for matched runs (seeds 43, 44): val_f1_valid:** Verified from training runner config and training_history.csv (column header val_f1_valid).
