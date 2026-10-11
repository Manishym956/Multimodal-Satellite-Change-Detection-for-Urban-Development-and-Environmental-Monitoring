# Paper-Drafting Foundation: Multimodal Satellite Change Detection for Urban Development

**Target Paper Title:**  
*Multimodal Satellite Change Detection for Urban Development: Evaluating Historical Optical Guidance under Asynchronous SAR–Optical Observations*

**Document Version:** 1.0 (Phase 3A Evidence Audit & Paper-Drafting Foundation)  
**Freeze Date:** October 11, 2026  
**Final Submission Deadline:** October 30, 2026 (19 days remaining)  
**Author / Project:** OJT Research Project  
**Repository Path:** `projects/multimodal-satellite-change-detection`  
**Integrity Status:** Read-only audit complete. All 18 model checkpoints, configuration manifests, dataset split definitions, training CSV logs, and metrics JSON files are frozen and cryptographically verified.

---

## 1. Research Scope and Research Questions

### 1.1 Context and Motivation
Urban change detection using Earth observation satellites is essential for infrastructure monitoring, informal settlement tracking, and municipal planning. While optical imagery (e.g., Sentinel-2) provides rich semantic and spectral content, it is frequently impeded by cloud cover, atmospheric haze, and solar illumination constraints. Conversely, Synthetic Aperture Radar (SAR, e.g., Sentinel-1) provides day-and-night, all-weather sensing capable of penetrating cloud cover, but suffers from speckle noise, geometric distortions (layover, foreshortening, shadow), and ambiguity in surface roughness vs. structural changes.

This study investigates **asynchronous multimodal change detection**: leveraging available historical pre-change optical imagery (T1) to guide change detection between bitemporal SAR observations (T1 and T2) when post-change optical imagery (T2) is unavailable due to adverse atmospheric conditions.

### 1.2 Formal Research Questions (RQs) and Scope Boundaries

| Research Question | Scientific Scope & Status | Bounded Conclusion / Classification |
|---|---|---|
| **RQ1: Historical Optical Guidance Utility**<br>*Can historical Sentinel-2 optical imagery improve Sentinel-1 SAR-based change detection when contemporaneous post-change optical imagery is unavailable?* | **Partially investigated**<br>Supported on OSCD; not demonstrated on SpaceNet 7. | **OSCD:** Clean Asynchronous CNN achieves test F1 = **0.4117** vs. SAR-only baseline **0.3558** (+0.0559 gain) and Naive Zero-Filled Async **0.2052**. Cross-Attention Transformer achieves **0.3564**.<br>**SpaceNet 7:** Synchronous CNN achieves test F1 = **0.1102** (Seed 42) / **0.1327** (Seeds 43/44), while Asynchronous CNN achieves test F1 = **0.0794** (Seed 42). Under extreme building label sparsity, historical optical guidance alone is insufficient to match synchronous bitemporal optical features. |
| **RQ2: Temporal Gap Sensitivity**<br>*How does the temporal separation between historical optical imagery and target SAR observations affect performance?* | **Not quantitatively established**<br>Dataset limitation; designated for future work. | **OSCD:** SAR images are temporal composites without precise individual acquisition timestamps; temporal-gap sensitivity cannot be evaluated.<br>**SpaceNet 7:** Evaluates strictly one fixed temporal pair (March 2018 to December 2019, a uniform 21-month separation across all 28 sites). No multi-interval ablation was conducted. Must be documented as a limitation. |
| **RQ3: SAR Dual-Polarization Contribution**<br>*Can additional SAR polarization channels (VV + VH) improve discrimination between structural changes and environmental pseudo-changes?* | **Not quantitatively established**<br>Single-polarization only; designated for future work. | All experiments in Scenario 1 and Scenario 2 utilized the available single VV channel (Scenario 1) or locked VV+VH configuration without an isolated single-pol vs. dual-pol ablation study. |
| **RQ4: Comparative Baseline Hierarchy**<br>*How does the proposed asynchronous multimodal approach compare against unimodal, early fusion, and dual-stream architectures?* | **Answered by existing evidence**<br>Established quantitatively on OSCD and SpaceNet 7. | **OSCD Hierarchy (Official test cities):**<br>Clean Async Dual-Stream CNN (0.4117) > Sync Dual-Stream CNN (0.3868) > Transformer Cross-Attention (0.3564) > SAR-only U-Net (0.3558) > Early Sync Fusion U-Net (0.3520) > Optical-only U-Net (0.2663) > Naive Async CNN (0.2052) > Transformer Self-Attention (0.1615).<br>**SpaceNet 7 Hierarchy (Locked test sites):**<br>Sync Dual-Stream CNN (0.1102–0.1327) > Async Dual-Stream CNN (0.0794–0.0813) > Cross-Attention Transformer (0.0412–0.0506). |
| **RQ5: Operating Conditions & Failure Modes**<br>*Under what conditions does historical optical guidance provide benefit, and where does it fail?* | **Partially investigated**<br>Demonstrated via error analysis and site inspections. | Historical optical guidance provides substantial noise reduction in stable urban footprints by constraining SAR false alarms over bare soil or seasonal roughness changes. However, it fails in dense construction boom zones where historical optical features are obsolete, and in unmasked cloud/sensor-boundary artifacts (e.g., SpaceNet 7 site `L15-1289E`). Environmental and agricultural conditions remain unverified because labels are strictly building-change proxies. |

---

## 2. Structured Paper Outline

1. **Title:** *Multimodal Satellite Change Detection for Urban Development: Evaluating Historical Optical Guidance under Asynchronous SAR–Optical Observations*
2. **Abstract:**
   - Problem: Cloud cover impedes optical change detection; SAR operates all-weather but suffers from speckle and roughness ambiguity.
   - Core premise: Asynchronous fusion using historical pre-change optical guidance to augment bitemporal SAR change detection.
   - Empirical findings: On OSCD benchmark, clean asynchronous CNN achieves test F1 = 0.4117, outperforming SAR-only (0.3558) and naive zero-filling (0.2052). On SpaceNet 7 building change proxies, synchronous optical retains a substantial margin over asynchronous models (F1 = 0.1327 vs. 0.0794).
   - Methodological rigour: Matched multi-seed analysis (seeds 43 and 44) resolves single-seed epoch confounds, demonstrating that masked loss does not alter the all-pixel F1 ceiling ($0.1247 \pm 0.0111$ vs. $0.1252 \pm 0.0130$) but provides valid-pixel constraint and boundary protection.
3. **Introduction:**
   - Urgent need for rapid urban change mapping.
   - Atmospheric vulnerability of optical earth observation.
   - The asynchronous observation paradigm: exploiting asymmetric temporal availability.
   - Summary of contributions and paper organization.
4. **Background and Related Work:**
   - Bitemporal optical change detection (Siamese networks, cross-attention).
   - SAR-based change detection and coherent/incoherent analysis.
   - Multimodal SAR–optical fusion (early fusion, late fusion, dual-stream architectures).
   - Addressing missing modalities and asynchronous satellite sensing.
5. **Study Design and Methodology:**
   - Dataset descriptions: OSCD (14 train / 10 test cities) and SpaceNet 7-derived dataset (20 train / 4 val / 4 test locked sites).
   - Temporal pairing and label formulation (OSCD direct annotations vs. SpaceNet 7 building proxy derivations).
   - Model Architectures: Dual-Stream U-Net (Sync, Clean Async, Naive Async) and Transformer Cross-Attention.
   - Optimization, class-weighting formulations (`pos_weight = 41.0` and `94.0`), and loss masking (`MaskedBCEWithLogitsLoss`).
   - Evaluation protocol: All-Pixel vs. Valid-Pixel metrics, primary fixed threshold $\tau = 0.50$, and validation-calibrated threshold $\tau^*$.
6. **Results:**
   - Scenario 1 (OSCD): Full baseline comparison, unimodal vs. multimodal, async vs. sync.
   - Scenario 2 (SpaceNet 7): Baseline model hierarchy under extreme label sparsity.
   - Experiment S2-1: Masked loss investigation and checkpoint selection ablation.
   - Matched Multi-Seed Experiment: Paired evaluation across seeds 43 and 44 (controlling weights, schedules, and selection rules).
7. **Discussion:**
   - Why historical optical guidance succeeds on OSCD but struggles on SpaceNet 7 (semantic context vs. high-resolution building footprint sparsity).
   - The impact of threshold calibration on extreme class imbalance.
   - Resolution of the epoch-progression confound in masked-loss training.
8. **Threats to Validity and Limitations:**
   - OSCD SAR compositing and lack of exact timestamps.
   - SpaceNet 7 building probability proxy labels (not direct multi-class change truth).
   - Single-interval evaluation (uniform 21-month gap).
   - Validation-to-test cross-city distribution shift.
9. **Future Work:**
   - Multi-temporal gap scaling experiments with precisely timestamped SAR tracks.
   - Integration of dual-polarization SAR (VV + VH) and interferometric coherence.
   - Direct multi-class semantic change detection benchmarks.
10. **Conclusion:** Summary of findings and engineering recommendations.
11. **References.**

---

## 3. Verified Dataset, Split, Preprocessing, and Model Specifications

### 3.1 Scenario 1: OSCD Benchmark
- **Dataset:** Onera Satellite Change Detection (OSCD) dataset.
- **Geographic Split:** Official OSCD partition:
  - **Train Cities (14):** `bordeaux`, `nantes`, `rennes`, `saclay_e`, `abudhabi`, `cupertino`, `pisa`, `hongkong`, `beirut`, `mumbai` (used in training); plus 4 validation cities: `beihai`, `bercy`, `aguasclaras`, `paris`.
  - **Test Cities (10):** `brasilia`, `montpellier`, `norcia`, `rio`, `saclay_w`, `valencia`, `dubai`, `lasvegas`, `milano`, `chongqing`.
- **Inputs & Modalities:**
  - Sentinel-1 SAR: VV polarization composite at T1 and T2 (2 channels).
  - Sentinel-2 Optical: 13 spectral bands (B01–B12, B8A). In asynchronous models, T1 optical is used (13 channels) and T2 optical is strictly withheld.
- **Preprocessing:** Normalization to $[0, 1]$; random cropping to $32 \times 32$ with 64 crops per city.
- **Class Imbalance & Loss:** BCE with `pos_weight = 41.0` (derived from training change ratio: $126,312 / 5,146,327 \approx 1:40.7$).
- **Optimizer:** Adam ($\text{lr} = 10^{-4}$), batch size 64, 50 epochs.
- **Checkpoint Selection:** Highest validation F1 on full validation-city scenes.

### 3.2 Scenario 2: SpaceNet 7-Derived Dataset
- **Dataset:** SEN12-MT Urban Mapping dataset (derived from SpaceNet 7 sites).
- **Locked Geographic Split:** Strategy B locked in `experiments/scenario_2/configs/split_definition.json` (`sha256: 5d0841adf0c6...`):
  - **Train Sites (20):** `L15-0331E-1257N`, `L15-0357E-1223N`, `L15-0358E-1220N`, `L15-0368E-1245N`, `L15-0434E-1218N`, `L15-0506E-1204N`, `L15-0544E-1228N`, `L15-0632E-0892N`, `L15-1138E-1216N`, `L15-1204E-1204N`, `L15-1209E-1113N`, `L15-1276E-1107N`, `L15-1296E-1198N`, `L15-1335E-1166N`, `L15-1389E-1284N`, `L15-1439E-1134N`, `L15-1617E-1207N`, `L15-1669E-1153N`, `L15-1672E-1207N`, `L15-1703E-1219N`.
  - **Validation Sites (4):** `L15-0457E-1135N`, `L15-0487E-1246N`, `L15-0977E-1187N`, `L15-1716E-1211N`.
  - **Test Sites (4):** `L15-0924E-1108N`, `L15-1289E-1169N`, `L15-1298E-1322N`, `L15-1438E-1134N`.
- **Temporal Pairing:** Fixed March 2018 (T1) to December 2019 (T2) (uniform 21-month gap across all sites).
- **Label Derivation:** Binary change proxy derived from thresholded building probability maps:
  $$\text{Label} = (\text{Building}_{T2} \ge 0.50) \oplus (\text{Building}_{T1} \ge 0.50)$$
  *Crucial note: This is a building footprint change proxy, not a direct multi-class land-use annotation.*
- **Inputs & Channels:**
  - S1 SAR: VV + VH at T1 and T2 (4 channels). NaNs filled to 0.0.
  - S2 Optical: 10 bands (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12).
  - Total Channels: Asynchronous = 14 ch (4 SAR + 10 Optical T1); Synchronous = 24 ch (4 SAR + 20 Optical T1+T2).
- **Class Imbalance & Loss:** BCE with `pos_weight = 94.0` (ratio: $4,718,348 / 50,387 = 93.64 \to 94.0$).
- **Optimizer:** Adam ($\text{lr} = 10^{-4}$), batch size 64, crop size 32, 64 crops per site (1,280 crops/epoch), 50 epochs.

---

## 4. Frozen Result Tables with Source Artifact Paths

### 4.1 Table 1: Scenario 1 (OSCD Benchmark, Official 10 Test Cities)
*Source Artifacts: `experiments/{experiment_id}/metrics.json` and `results/tables/final_model_comparison.csv`.*  
*Evaluation protocol: Full scene evaluation at $\tau = 0.50$ on pooled test city pixels.*

| Model Name | Input Modalities | Parameters | Val Best Ep | Val F1 | Test Prec | Test Rec | Test F1 | Test IoU | Test FAR | TP | FP | FN | TN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **SAR-only U-Net** | S1 VV (T1+T2) | 31,036,545 | 21 | 0.0662 | 0.4558 | 0.2917 | **0.3558** | 0.2164 | 0.0190 | 46,407 | 55,401 | 112,670 | 2,863,458 |
| **Optical-only U-Net** | S2 Optical (T1+T2) | 31,051,969 | 50 | 0.2387 | 0.2515 | 0.2830 | **0.2663** | 0.1536 | 0.0459 | 45,025 | 134,013 | 114,052 | 2,784,846 |
| **Sync Fusion U-Net** | S1 + S2 Early Fusion | 31,052,033 | 46 | 0.2538 | 0.2877 | 0.4533 | **0.3520** | 0.2136 | 0.0612 | 72,104 | 178,489 | 86,973 | 2,740,370 |
| **Sync Dual-Stream CNN** | S1 (T1+T2) + S2 (T1+T2) | 27,284,097 | 31 | 0.3523 | 0.3009 | 0.5415 | **0.3868** | 0.2398 | 0.0686 | 86,137 | 200,124 | 72,940 | 2,718,735 |
| **Naive Async CNN** | S1 (T1+T2) + S2 (T1+Zero T2) | 27,284,097 | 46 | 0.1093 | 0.2363 | 0.1814 | **0.2052** | 0.1143 | 0.0319 | 28,849 | 93,242 | 130,228 | 2,825,617 |
| **Clean Async CNN** | S1 (T1+T2) + S2 Hist T1 | 27,276,609 | 26 | 0.0992 | 0.4753 | 0.3631 | **0.4117** | 0.2592 | 0.0218 | 57,764 | 63,762 | 101,313 | 2,855,097 |
| **Transformer Self-Attn** | S1 (T1+T2) + S2 Hist T1 | 1,358,081 | 37 | 0.1333 | 0.0974 | 0.4734 | **0.1615** | 0.0879 | 0.2391 | 75,307 | 697,994 | 83,770 | 2,220,865 |
| **Transformer Cross-Attn** | S1 Q + S2 Hist T1 K,V | 1,156,993 | 33 | 0.1428 | 0.2615 | 0.5592 | **0.3564** | 0.2168 | 0.0861 | 88,959 | 251,196 | 70,118 | 2,667,663 |

---

### 4.2 Table 2: Scenario 2 Initial Baselines & Masked-Loss Ablations (Seed 42)
*Source Artifacts: `experiments/scenario_2/{model}/metrics.json` and `experiments/scenario_2_masked_loss/{model}/metrics.json`.*  
*Evaluation protocol: Evaluated on 4 locked test sites at $\tau = 0.50$.*

| Experiment & Model | Loss Formulation | Checkpoint Rule | Sel. Ep | Val F1 | Test Prec (all) | Test Rec (all) | Test F1 (all) | Test IoU (all) | Test FAR (all) | Test F1 (valid) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Async CNN Baseline** | Standard BCE | `val_f1_all` | 2 | 0.1832 | 0.0479 | 0.2334 | **0.0794** | 0.0414 | 0.0336 | 0.0804 |
| **Sync CNN Baseline** | Standard BCE | `val_f1_all` | 21 | 0.2663 | 0.0610 | 0.5680 | **0.1102** | 0.0583 | 0.0633 | 0.1102 |
| **Cross-Attn Baseline** | Standard BCE | `val_f1_all` | 23 | 0.1328 | 0.0214 | 0.5705 | **0.0412** | 0.0210 | 0.1892 | 0.0412 |
| **Async CNN Masked** | Masked BCE | `val_f1_valid` | 2 | 0.1677 | 0.0473 | 0.2907 | **0.0813** | 0.0424 | 0.0424 | 0.0822 |
| **Sync CNN Masked** | Masked BCE | `val_f1_valid` | 30 | 0.2878 | 0.0781 | 0.5342 | **0.1363** | 0.0731 | 0.0456 | 0.1376 |
| **Cross-Attn Masked** | Masked BCE | `val_f1_valid` | 41 | 0.2223 | 0.0267 | 0.4639 | **0.0506** | 0.0259 | 0.1222 | 0.0506 |

---

## 5. Matched Multi-Seed Sync CNN Results and Conservative Interpretation

### 5.1 Experimental Setup & Integrity Controls
To eliminate the epoch-progression confound and evaluate whether Masked BCE provides a genuine statistical gain, a controlled experiment was run across Seeds 43 and 44 using the unified runner `src/training/train_scenario2_sync_multiseed.py`:
- **Paired Initialization:** Exact state dictionary match within each seed pair ($\Delta = 0.0$).
- **Paired Sampling:** Pre-generated deterministic crop schedules matching exact site dimensions, ensuring standard BCE and masked BCE consumed identical crops in identical order.
- **Predeclared Selection Criterion:** Highest Valid-Pixel Validation F1 (`val_f1_valid`).

### 5.2 Primary Matched Comparison Table (Seeds 43 and 44)
*Source Artifacts: `experiments/scenario_2_multiseed/sync_cnn_{baseline|masked}_seed{43|44}/metrics.json`.*

| Run Identifier | Loss Formulation | Seed | Best Ep | Val F1 (valid) | Test Prec (all) | Test Rec (all) | Test F1 (all @ 0.50) | Test FAR (all) | Test F1 (valid) | Calibrated $\tau^*$ | Test F1 (all @ $\tau^*$) |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sync_cnn_baseline_seed43` | Standard BCE | 43 | 30 | 0.2932 | 0.0757 | 0.5349 | **0.1327** | 0.0472 | 0.1342 | 0.79 | 0.1943 |
| `sync_cnn_masked_seed43` | Masked BCE | 43 | 30 | 0.2877 | 0.0693 | 0.5719 | **0.1236** | 0.0556 | 0.1257 | 0.83 | 0.1912 |
| `sync_cnn_baseline_seed44` | Standard BCE | 44 | 43 | 0.2746 | 0.0751 | 0.5721 | **0.1327** | 0.0510 | 0.1337 | 0.86 | 0.2180 |
| `sync_cnn_masked_seed44` | Masked BCE | 44 | 43 | 0.2744 | 0.0636 | 0.5572 | **0.1141** | 0.0594 | 0.1180 | 0.91 | 0.2113 |

### 5.3 Descriptive Statistical Aggregation

#### Primary Matched Analysis (Seeds 43 & 44)
- **Baseline (Standard BCE) All-Pixel F1:**
  - Seed 43: $0.1327$ | Seed 44: $0.1327$
  - Descriptive Mean $\pm$ Sample Std: $\mathbf{0.1327 \pm 0.0000}$
- **Masked BCE All-Pixel F1:**
  - Seed 43: $0.1236$ | Seed 44: $0.1141$
  - Descriptive Mean $\pm$ Sample Std: $\mathbf{0.1188 \pm 0.0067}$
- **Pairwise Differences ($\Delta = \text{Masked} - \text{Base}$):**
  - Seed 43: $\Delta = -0.0091$
  - Seed 44: $\Delta = -0.0186$
  - Mean Paired Difference: $\mathbf{-0.0139}$

#### Historical 3-Seed Context (Including Seed 42)
*Note: Seed 42 is reported for historical transparency; its baseline used all-pixel validation selection, while Seeds 43 and 44 used valid-pixel validation selection.*
- **Baseline F1 (Seeds 42, 43, 44):** $0.1102, 0.1327, 0.1327 \to \text{Mean} = \mathbf{0.1252 \pm 0.0130}$
- **Masked F1 (Seeds 42, 43, 44):** $0.1363, 0.1236, 0.1141 \to \text{Mean} = \mathbf{0.1247 \pm 0.0111}$
- **3-Seed Mean Difference:** $\mathbf{-0.0005}$

### 5.4 Conservative Scientific Interpretation
1. **Descriptive Nature of the Findings:**
   With two matched seeds ($N=2$) and three historical seeds ($N=3$), these figures are **descriptive comparative metrics**, not statistically significant proofs of superiority or inferiority.
2. **Resolution of the Single-Seed Anomaly:**
   In Seed 42, masked loss appeared to improve F1 by $+0.0261$. In the strictly controlled Seed 43 and Seed 44 runs, where initial weights and spatial crop sampling schedules were identical within each pair, masked loss showed slight negative differences ($-0.0091$ and $-0.0186$). Across all three seeds, the overall mean difference is negligible ($\Delta = -0.0005$).
3. **Epoch Tracking Confirmation:**
   When models are trained on identical crop schedules, baseline and masked loss models reach their peak validation F1 at the **exact same epoch** (Epoch 30 in Seed 43, Epoch 43 in Seed 44). This directly confirms that the apparent gain in Seed 42 was confounded by epoch progression (Epoch 30 vs. Epoch 21).
4. **Role of Masked Loss:**
   Masked loss does not lift the global test F1 ceiling on SpaceNet 7, but it remains mathematically necessary to prevent invalid nodata SAR pixels (e.g., site `L15-1289E`) from corrupting the gradient during backpropagation.
5. **Operating Point (Threshold) Analysis:**
   Validation-calibrated thresholding ($\tau^* \in [0.78, 0.91]$) shifts the operating point to reduce false alarms under extreme imbalance (`pos_weight = 94.0`), elevating test F1 to $\sim 0.19\text{--}0.22$. However, this is an operating-point shift, not evidence of improved model calibration stability.

---

## 6. Required Figures and Tables for the Final Paper

### 6.1 Existing Verified Figures
Located in `results/figures/` and `results/qualitative/`:
- `results/figures/f1_comparison.pdf`: OSCD model F1 bar chart.
- `results/figures/precision_recall_comparison.pdf`: Precision vs. Recall across OSCD models.
- `results/figures/far_comparison.pdf`: False Alarm Rate across OSCD models.
- `results/figures/confusion_matrix_comparison.pdf`: Confusion matrix panels.
- `results/qualitative/example_01/montpellier_panel.png`: Qualitative comparison on Montpellier.
- `results/qualitative/example_02/dubai_panel.png`: Qualitative comparison on Dubai.

### 6.2 New Figures and Tables to Produce for Phase 3B
1. **Figure 1: Dual-Stream Asynchronous Architecture Diagram**
   - Flow diagram contrasting the Clean Asynchronous dual-stream model (S2 historical optical only) against the Synchronous model (bitemporal optical).
2. **Figure 2: Multi-Seed Training Trajectory Plot**
   - Epoch-by-epoch validation F1 curves comparing Standard BCE vs. Masked BCE across Seeds 43 and 44, visualizing the identical peak epoch alignment (Epoch 30 and 43).
3. **Figure 3: Threshold Sensitivity Curve ($\tau$ vs. Precision, Recall, F1)**
   - Illustrating the operating-point shift between default $\tau = 0.50$ and validation-calibrated $\tau^*$ under severe class imbalance (`pos_weight = 94.0`).
4. **Figure 4: Qualitative SpaceNet 7 Error Map**
   - Highlighting site `L15-1289E` showing how invalid SAR boundary regions generate false alarms if unmasked.

---

## 7. Limitations and Threats to Validity

1. **OSCD Temporal Composite Limitation:**
   OSCD SAR imagery consists of temporal composites without exact per-acquisition timestamps. Consequently, OSCD cannot evaluate temporal-gap sensitivity (RQ2).
2. **SpaceNet 7 Single Fixed Interval:**
   Scenario 2 evaluates exclusively a single fixed interval (March 2018 to December 2019, 21 months). Generalization across varying temporal separations remains unverified.
3. **Derived Building Proxy Labels:**
   SpaceNet 7 change masks are derived proxies obtained by thresholding bitemporal building probability maps ($p \ge 0.50$). They do not represent direct ground-truth survey annotations for general environmental change, deforestation, or agriculture.
4. **Geographic Cross-City Domain Shift:**
   In OSCD, Clean Async CNN achieved validation F1 = $0.0992$ but test F1 = $0.4117$. This substantial variance reflects significant radiometric and morphological domain divergence across geographic regions.
5. **Sample Size of Random Seeds:**
   While the matched multi-seed experiment resolves the single-seed confound for Sync CNN, computational constraints limited the evaluation to two matched seeds ($43, 44$) and one historical seed ($42$). Results must be reported as descriptive empirical findings.

---

## 8. Reproducibility Checklist

- [x] Locked split definition preserved and verified (`sha256: 5d0841adf0c6...`).
- [x] All 18 model checkpoints (`best.pt`) present on disk with recorded SHA256 hashes.
- [x] Deterministic crop schedule generator verified against exact training site dimensions.
- [x] Model initialization state dictionaries matched with zero difference ($\Delta = 0.0$) within each seed pair.
- [x] DataLoader batch identity confirmed between paired standard and masked conditions.
- [x] Metrics recomputed and verified against raw confusion counts ($TP, FP, FN, TN$).
- [x] Predeclared checkpoint selection rule (`val_f1_valid`) consistently applied in matched runs.
- [x] Output paths completely isolated under `scenario_2_multiseed/`.

---

## 9. Unresolved Discrepancies and Missing Evidence Log

1. **Seed 42 Baseline Selection Metric:**
   In Seed 42, the baseline model selected `best.pt` using `val_f1_all` (All-Pixel), whereas masked loss used `val_f1_valid` (Valid-Pixel). In Seeds 43 and 44, both used `val_f1_valid`. This procedural difference is documented, and Seed 42 is preserved as historical context rather than a direct matched pair.
2. **Unsaved Intermediate Weights for S2-1 Async CNN Epoch 18:**
   In S2-1 Async CNN, all-pixel validation F1 peaked at Epoch 18 ($0.1600$), but disk checkpoints were restricted to `best.pt` (Epoch 2) and `last.pt` (Epoch 50). Per strict preservation rules, Epoch 18 weights were not reconstructed or evaluated.
3. **Dual-Polarization (VV+VH) Ablation:**
   Neither dataset includes an isolated single-polarization vs. dual-polarization ablation under identical architectures. RQ3 must be designated as future work.
