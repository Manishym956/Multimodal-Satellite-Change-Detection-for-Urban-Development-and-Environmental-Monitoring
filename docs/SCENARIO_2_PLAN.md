# Scenario 2 Plan — SEN12-MT Urban Mapping Dataset

Status: SPLIT LOCKED — READY FOR LOADER IMPLEMENTATION
Last updated: 2026-10-06

---

## Dataset confirmed

**Dataset:** SEN12-MT Urban Mapping (SpaceNet 7 sites)
**Root:** `datasets/sen12_mt_urbanmapping/multimodal_cd_dataset/`
**Sites:** 80 geographic sites
**Timestamps:** 18–26 per site (monthly, 2017–2020)
**Size on disk:** 18.45 GB

See `docs/SCENARIO_2_DATASET_INSPECTION.md` for the full audit.

---

## Research question

Does the performance behaviour observed on the small OSCD benchmark (Scenario 1) persist when asynchronous SAR-optical change detection is evaluated using the larger and more diverse SEN12-MT Urban Mapping dataset?

Specifically:
- Does the Clean Asynchronous Dual-Stream CNN remain the strongest architecture?
- Does the SAR-guided Cross-Attention Transformer close the gap with the CNN?
- Does VH polarization add useful signal over VV-only?

---

## Hypothesis (pre-registered before training)

1. CNN advantage over transformers observed on OSCD may diminish with more training data.
2. The asynchronous approach (historical optical + SAR) will remain competitive with synchronous.
3. VV+VH SAR is expected to improve over VV-only, but the magnitude is unknown.

---

## Intended models

| # | Model | Source file | Scenario 2 role |
|---|---|---|---|
| S2-1 | Synchronous Dual-Stream CNN | `dual_stream_unet.py` | Synchronous baseline |
| S2-2 | Clean Async Dual-Stream CNN | `dual_stream_unet_clean.py` | Scenario 1 winner |
| S2-3 | SAR-guided Cross-Attention Transformer | `transformer_cross_attention.py` | Main attention model |

No new architectures. The Self-Attention Transformer is excluded (underperformed Scenario 1).

---

## Input modalities (VERIFIED from dataset inspection)

| Channel | Scenario 1 | Scenario 2 |
|---|---|---|
| SAR T1 | VV only (1 ch) | VV + VH (2 ch) |
| SAR T2 | VV only (1 ch) | VV + VH (2 ch) |
| Historical optical | S2 13 bands (T1) | S2 10 bands (T1 only) |
| Contemporary optical T2 | NOT USED async | NOT USED async |

Total input channels:
- Async models: 4 SAR + 10 optical = **14 channels**
- Sync baseline: 4 SAR + 10 optical T1 + 10 optical T2 = **24 channels**

Key differences from Scenario 1:
- SAR is 4 channels (VV+VH both dates), not 2 (VV only)
- Optical is 10 bands, not 13
- S2 is already in [0,1], no divide by 10000

---

## Temporal pairing strategy — CONFIRMED: Strategy B

**Fixed pair: T1 = 2018-03 (March 2018), T2 = 2019-12 (December 2019)**
**Temporal interval: 21 months (uniform across all sites)**

Full analysis in `docs/SCENARIO_2_TEMPORAL_PAIRING_ANALYSIS.md`.

| Eligibility | Count |
|---|---|
| Common eligible sites (all 3 models) | **28** |
| Async-only (excluded — S2@T2 missing) | 4 |
| Zero-positive-pixel sites | 0 |

The synchronous baseline requires S2 at T2, which is the binding constraint.
All three models are evaluated on the same **28 common sites**.

---

## Target prediction task

Binary urban building change detection (derived label):
- **Label = (buildings_2019-12 ≥ 0.5) XOR (buildings_2018-03 ≥ 0.5)**
- Positive class = pixel changed between non-building and building (or vice versa)
- Change fraction: min 0.044%, max 4.225%, mean 1.005%, median 0.733% across 28 sites
- No sites have zero positive pixels at threshold 0.5
- This is a proxy label, not a directly annotated change mask. Must be stated in paper.

---

## Temporal design for asynchronous models

| Input | Description | Timestamp |
|---|---|---|
| SAR T1 | S1 VV+VH | 2018-03 |
| Historical optical | S2 10 bands | 2018-03 |
| SAR T2 | S1 VV+VH | 2019-12 |
| Contemporary optical T2 | S2 10 bands at T2 | **WITHHELD from async models** |
| Change label | (bld_2019-12 ≥ 0.5) XOR (bld_2018-03 ≥ 0.5) | — |

The synchronous baseline receives S2 at both 2018-03 and 2019-12.

---

## Train / validation / test split — LOCKED

**Split file:** `experiments/scenario_2/configs/split_definition.json`
**Common eligible sites:** 28 (synchronous constraint is binding)
**Seed:** 42

| Split | Sites | Pixels | Change pixels | Change % |
|---|---|---|---|---|
| Train | 20 | 4,768,735 | 50,387 | 1.057% |
| Validation | 4 | 955,017 | 9,828 | 1.029% |
| Test | 4 | 954,528 | 6,859 | 0.719% |

**pos_weight (from training sites only): 93.6 → use 94 in BCEWithLogitsLoss**

Split separation verified: no overlaps. This split must not be modified after training begins.

**Limitation:** The test set is 4 sites (~955K pixels), approximately 31% of the OSCD test set volume. Results should be treated as indicative. This must be disclosed in the paper.

---

## Preprocessing

| Step | Detail |
|---|---|
| S1 NaN handling | Fill with 0.0 before tensor conversion |
| S1 value range | [0, 1] (already normalised in dataset) |
| S2 value range | [0, 1] (already normalised; do NOT divide by 10000) |
| Buildings threshold | 0.5 for binary change mask |
| Crop size | 32 × 32 (same as Scenario 1; sites are ~489×488 so sufficient) |
| Crops per site per epoch | 64 (same as Scenario 1) |
| Augmentation | None (same as Scenario 1) |

---

## Class imbalance

Training split change fraction: **1.057%** (50,387 / 4,768,735 pixels).
**pos_weight = 93.6 (rounded to 94)** — computed from the 20 training sites.
This is approximately twice the OSCD value (41), reflecting the sparser change class.
Do not use the OSCD pos_weight for SEN12-MT.

---

## Evaluation metrics

Same as Scenario 1:
- Precision, Recall, F1, IoU, FAR
- Pooled across test sites
- Per-site mean ± std reported separately
- Undefined rates (zero denominator) reported as undefined, not zero
- Threshold: 0.5

---

## Reproducibility

- Seed: 42
- PyTorch: 2.5.1+cu124
- GPU: NVIDIA RTX 4060 Laptop GPU (8 GB)
- Optimizer: Adam
- LR: 1e-4 (starting point; may be updated by sweep)
- Batch size: 64
- Epochs: 50 (may extend to 100 if not converged)
- Checkpoint: best validation F1; last.pt saved every epoch

---

## Risks and limitations

1. **Change label is derived, not directly annotated.** The threshold (0.5) and T1/T2 selection must be stated in the paper.
2. **Sparse change class.** 1.057% change pixels in training; pos_weight=94 required.
3. **Small test set.** 4 sites, ~955K pixels, ~31% of OSCD test volume. Results are indicative, not definitive. Must be disclosed.
4. **No official split.** Fixed seed-42 site-level split. Locked in `split_definition.json`.
5. **Different modalities from Scenario 1.** 4-ch SAR and 10-band S2 vs 2-ch SAR and 13-band S2. Cross-scenario absolute F1 comparison is not valid.
6. **NaN in S1.** Bad timestep indices documented; NaN filled with 0.0.
7. **Single seed.** Variance across seeds not estimated.

---

## Comparison to Scenario 1

Absolute F1 scores between Scenario 1 (OSCD) and Scenario 2 (SEN12-MT) must NOT be directly compared. Valid comparison is the relative architecture ranking within each scenario.
