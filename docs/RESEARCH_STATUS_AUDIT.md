# Research Status Audit and Experiment Inventory

**Audit Date:** October 11, 2026  
**Final Project Deadline:** October 30, 2026 (19 days remaining)  
**Project Title:** Multimodal Satellite Change Detection for Urban Development and Environmental Monitoring  
**Target Repository:** projects/multimodal-satellite-change-detection  
**Integrity Status:** All 14 baseline and masked-loss checkpoints, locked split definitions, metrics JSON files, and training CSV histories verified intact.

---

## 1. Executive Summary

This document establishes the verified source of truth for all experimental artifacts across Scenario 1 (OSCD) and Scenario 2 (SpaceNet 7-derived dataset), audits experiment integrity, identifies methodological confounds, reconciles the Research Requirement Document (RRD) research questions, and outlines a prioritized roadmap for the remaining 19 days.

### Core Audit Findings
1. **Repository & Artifact Integrity:**
   - The locked split definition (experiments/scenario_2/configs/split_definition.json) remains completely unchanged (sha256: 5d0841adf0...).
   - All 14 primary checkpoints (8 for OSCD, 3 for Scenario 2 Baseline, 3 for Scenario 2 Masked Loss) exist on disk, are non-empty, and their SHA256 hashes are cataloged in experiments/research_status/artifact_integrity.json.
   - All precision, recall, F1, IoU, and false alarm rate (FAR) metrics reported in JSON and Markdown were recomputed directly from raw confusion counts (, FP, FN, TN$) and match within floating-point tolerance ($< 10^{-4}$).
2. **Key Methodological Limitations & Confounds Discovered:**
   - **OSCD SAR Temporal Gap Limitation:** OSCD SAR images are multi-temporal composites without precise acquisition timestamps. Consequently, OSCD cannot evaluate temporal-gap sensitivity (RQ2).
   - **Scenario 1 Val/Test Mismatch:** The Clean Async CNN achieved test F1 = 0.4117 on OSCD, but peaked at validation F1 = 0.0992 (Epoch 26). The severe gap between validation and test city distributions highlights significant cross-city domain shift.
   - **Scenario 2 Masked-Loss Epoch-Progression Confound:** Masked-loss training yielded higher all-pixel F1 for Sync CNN (0.1363 vs 0.1102) and Cross-Attention Transformer (0.0506 vs 0.0412). However, their checkpoints were selected at later epochs (Epoch 30 vs 21 for Sync CNN; Epoch 41 vs 23 for Transformer). These gains cannot yet be attributed purely to masked loss.
   - **Async CNN Checkpoint Selection Invariance:** For Async CNN, baseline and masked-loss models are both matched at Epoch 2 (F1: 0.0794 baseline vs 0.0813 masked loss). Masked loss had minimal effect on Async CNN performance. S2-1 Async CNN all-pixel validation F1 peaked at Epoch 18 (0.1600), but those weights were not saved to disk and must not be retrained.
   - **Semantic Scope Limitation:** SpaceNet 7 labels are binary building-detection change proxies (derived from building probability maps thresholded at 0.50). They do not represent agriculture, environmental degradation, or general land-cover shifts. General environmental monitoring claims cannot be made.

---

## 2. Experiment Artifact Inventory

Below is the verified inventory of all completed experiments. Raw records are archived in experiments/research_status/experiment_inventory.csv.

### Table 1: Scenario 1 (OSCD Benchmark, Official 14 Train / 10 Test Cities)
*All models evaluated at $\tau = 0.50$ on pooled test city pixels. Training loss: BCE with pos_weight=41.0 (except unimodal/early-fusion as noted).*

| Model | Modality | Best Val Ep | Val F1 | Test Prec | Test Rec | Test F1 | Test IoU | Test FAR | Test TP | Test FP | Test FN | Test TN | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **SAR-only U-Net** | S1 VV (T1+T2) | 21 | 0.0662 | 0.4558 | 0.2917 | **0.3558** | 0.2164 | 0.0190 | 46,407 | 55,401 | 112,670 | 2,863,458 | Verified |
| **Optical-only U-Net** | S2 Optical (T1+T2) | 50 | 0.2387 | 0.2515 | 0.2830 | **0.2663** | 0.1536 | 0.0459 | 45,025 | 134,013 | 114,052 | 2,784,846 | Verified |
| **Sync Fusion U-Net** | S1 + S2 Early Fusion | 46 | 0.2538 | 0.2877 | 0.4533 | **0.3520** | 0.2136 | 0.0612 | 72,104 | 178,489 | 86,973 | 2,740,370 | Verified |
| **Sync Dual-Stream CNN** | S1 (T1+T2) + S2 (T1+T2) | 31 | 0.3523 | 0.3009 | 0.5415 | **0.3868** | 0.2398 | 0.0686 | 86,137 | 200,124 | 72,940 | 2,718,735 | Verified |
| **Naive Async CNN** | S1 (T1+T2) + S2 (T1+Zero T2) | 46 | 0.1093 | 0.2363 | 0.1814 | **0.2052** | 0.1143 | 0.0319 | 28,849 | 93,242 | 130,228 | 2,825,617 | Verified |
| **Clean Async CNN** | S1 (T1+T2) + S2 Hist T1 (13ch) | 26 | 0.0992 | 0.4753 | 0.3631 | **0.4117** | 0.2592 | 0.0218 | 57,764 | 63,762 | 101,313 | 2,855,097 | Verified |
| **Transformer Self-Attn** | S1 (T1+T2) + S2 Hist T1 | 37 | 0.1333 | 0.0974 | 0.4734 | **0.1615** | 0.0879 | 0.2391 | 75,307 | 697,994 | 83,770 | 2,220,865 | Verified |
| **Transformer Cross-Attn** | S1 Q + S2 Hist T1 K,V | 33 | 0.1428 | 0.2615 | 0.5592 | **0.3564** | 0.2168 | 0.0861 | 88,959 | 251,196 | 70,118 | 2,667,663 | Verified |

---

### Table 2: Scenario 2 (SpaceNet 7-Derived Dataset, Locked 20 Train / 4 Val / 4 Test Sites)
*Pairing: March 2018 (T1) to December 2019 (T2). Loss pos_weight = 93.6. Threshold $\tau = 0.50$.*

| Experiment | Model | Loss Type | Sel. Rule | Best Ep | Val F1 | Test Prec | Test Rec | Test F1 | Test IoU | Test FAR | TP | FP | FN | TN |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **S2 Baseline** | Async Dual-Stream CNN | Standard BCE | all-pixel | 2 | 0.1832 | 0.0479 | 0.2334 | **0.0794** | 0.0414 | 0.0336 | 1,601 | 31,844 | 5,258 | 915,825 |
| **S2 Baseline** | Sync Dual-Stream CNN | Standard BCE | all-pixel | 21 | 0.2663 | 0.0610 | 0.5680 | **0.1102** | 0.0583 | 0.0633 | 3,896 | 59,965 | 2,963 | 887,704 |
| **S2 Baseline** | Cross-Attn Transformer | Standard BCE | all-pixel | 23 | 0.1328 | 0.0214 | 0.5705 | **0.0412** | 0.0210 | 0.1892 | 3,913 | 179,292 | 2,946 | 768,377 |
| **S2-1 Masked (All-Px)** | Async Dual-Stream CNN | Masked BCE | valid-pixel | 2 | 0.1677 | 0.0473 | 0.2907 | **0.0813** | 0.0424 | 0.0424 | 1,994 | 40,187 | 4,865 | 907,482 |
| **S2-1 Masked (All-Px)** | Sync Dual-Stream CNN | Masked BCE | valid-pixel | 30 | 0.2878 | 0.0781 | 0.5342 | **0.1363** | 0.0731 | 0.0456 | 3,664 | 43,241 | 3,195 | 904,428 |
| **S2-1 Masked (All-Px)** | Cross-Attn Transformer | Masked BCE | valid-pixel | 41 | 0.2223 | 0.0267 | 0.4639 | **0.0506** | 0.0259 | 0.1222 | 3,182 | 115,791 | 3,677 | 831,878 |
| **S2-1 Masked (Valid-Px)** | Async Dual-Stream CNN | Masked BCE | valid-pixel | 2 | 0.1677 | 0.0479 | 0.2895 | **0.0822** | 0.0428 | 0.0479 | 1,973 | 39,239 | 4,843 | 779,213 |
| **S2-1 Masked (Valid-Px)** | Sync Dual-Stream CNN | Masked BCE | valid-pixel | 30 | 0.2878 | 0.0790 | 0.5339 | **0.1376** | 0.0739 | 0.0519 | 3,639 | 42,438 | 3,177 | 776,014 |
| **S2-1 Masked (Valid-Px)** | Cross-Attn Transformer | Masked BCE | valid-pixel | 41 | 0.2223 | 0.0268 | 0.4630 | **0.0506** | 0.0260 | 0.1402 | 3,156 | 114,720 | 3,660 | 703,732 |

---

## 3. Discrepancy and Audit Log

1. **FAR Definition & Calculation Discrepancy:**
   - In earlier Scenario 2 baseline summaries, FAR was loosely reported as  / (FP + TN) = 0.0348$ for Async CNN baseline, while exact counts give ,844 / (31,844 + 915,825) = 0.03360$. Recomputed values from counts are now locked as authoritative.
2. **Checkpoint Selection Protocol Inconsistency:**
   - Scenario 2 baselines used al_f1 (all-pixel) on the validation set for est.pt selection.
   - S2-1 masked loss used al_f1_valid (valid-pixel) for est.pt selection.
   - Audit finding: For Sync CNN and Cross-Attention Transformer, both metrics peaked at the exact same epoch (Ep 30 and Ep 41 respectively). For Async CNN, al_f1_valid peaked at Ep 2, whereas al_f1_all peaked at Ep 18. However, because Ep 18 weights were not preserved on disk, Ep 2 remains the sole evaluated checkpoint.
3. **Epoch Progression Confound in S2-1 Gains:**
   - The reported improvement for Sync CNN (+0.0261 F1) and Cross-Attention (+0.0094 F1) cannot be disentangled from the fact that they trained 9 and 18 epochs longer than their respective baselines before peaking.

---

## 4. Reconciliation of RRD Research Questions

| RQ | Original Description | Current Status | Supporting Evidence & Honest Classification |
|:---:|:---|:---:|:---|
| **RQ1** | Can historical Sentinel-2 optical imagery improve Sentinel-1 change detection when synchronous optical is unavailable? | **Partially investigated** | **Supported on OSCD:** Clean Async CNN achieves test F1 = 0.4117 vs SAR-only 0.3558 (+0.0559 gain) and Naive Async 0.2052. Cross-Attention Transformer achieves 0.3564.<br>**Not demonstrated on SpaceNet 7:** Async CNN achieves test F1 = 0.0794 vs Sync CNN 0.1102. Historical optical guidance on SpaceNet 7 did not outperform synchronous optical, and extreme building label sparsity makes historical guidance alone insufficient. |
| **RQ2** | How does temporal gap between historical optical imagery and target SAR affect performance? | **Not quantitatively established** | **Cannot be answered with existing data.** OSCD SAR images are multi-temporal composites without acquisition dates. SpaceNet 7 experiments evaluated only a single fixed pair (March 2018 to December 2019, 21-month gap). Must be documented honestly as a dataset limitation and designated for future work. |
| **RQ3** | Can additional SAR information (VV and VH polarization) improve change discrimination? | **Not quantitatively established** | **All current experiments used single-polarization (VV only).** The codebase does not include dual-pol (VV+VH) ablation experiments. Must be reported as uninvestigated and left for future work. |
| **RQ4** | How does the proposed asynchronous multimodal approach compare with unimodal, conventional fusion, and dual-stream baselines? | **Answered by existing evidence** | **Fully benchmarked on OSCD:**<br>1. Clean Async CNN (0.4117) > Sync Dual-Stream (0.3868) > SAR-only (0.3558) > Sync Early Fusion (0.3520) > Optical-only (0.2663) > Naive Async (0.2052).<br>2. On SpaceNet 7, Sync CNN (0.1102) > Async CNN (0.0794) > Cross-Attention (0.0412). Complete quantitative baseline hierarchy established. |
| **RQ5** | Under what temporal and environmental conditions does historical optical guidance provide greatest benefit? | **Partially investigated** | **Investigated via error analysis and qualitative inspections:** Historical optical guidance reduces false alarms caused by surface roughness and moisture changes in urban areas with stable historical building footprints. However, severe degradation occurs in rapid construction sites and unmasked cloud/sensor-boundary regions (L15-1289E). Environmental/agricultural conditions are unverified due to building-only proxy labels. |

---

## 5. Matched Sync CNN Multi-Seed Experiment Proposal

To establish statistical robustness before the submission deadline, we evaluate whether to run a matched multi-seed experiment for Sync CNN (comparing Standard BCE vs Masked BCE across seeds 43 and 44).

### Experimental Setup & Controls
- **Model:** Sync Dual-Stream CNN (DualStreamUNetSync, 27,284,097 parameters).
- **Split:** Locked Strategy B (20 train, 4 val, 4 test sites from split_definition.json).
- **Input Channels:** SAR T1+T2 (2 ch) + Optical T1+T2 (6 ch) = 8 ch total.
- **Hyperparameters:** Batch size = 64, Crop size = 32, Crops/site = 64, LR = 1e-4 (Adam), Epochs = 50, Pos Weight = 93.6.
- **Experimental Variable:** Standard BCE Loss vs Masked BCE Loss (unmasked pixels only).
- **Checkpoint Selection Rule:** In S2-1, both al_f1_all and al_f1_valid peaked at the exact same epoch for Sync CNN (Ep 30). For seeds 43 and 44, both metrics must be logged, and selection must use the matched criterion (or track both) to prevent selection bias.
- **Evaluation:** Dual-protocol testing at $\tau = 0.50$ and calibrated $\tau^*$.

### Runtime and Resource Estimates
- Each Sync CNN 50-epoch run takes **~513 to 557 seconds** (~8.5 to 9.3 minutes) on the NVIDIA RTX 4060 Laptop GPU.
- Total training runs proposed: 4 runs (Seed 43 Baseline, Seed 43 Masked, Seed 44 Baseline, Seed 44 Masked).
- Total compute time: **~35 to 40 minutes**.
- **Conclusion:** Highly feasible and safe within the remaining 19 days, requiring minimal compute budget.

---

## 6. Recommended Task Order Through October 30

| Date Window | Priority / Task | Key Deliverables & Milestones |
|---|---|---|
| **Oct 11 (Today)** | **Phase 1: Freeze & Audit** | Establish artifact inventory, compute hashes, lock baseline records, freeze code. |
| **Oct 12 – Oct 13** | **Phase 2: Multi-Seed Experiment (Optional/Controlled)** | Execute 4 matched runs for Sync CNN (seeds 43, 44; Baseline vs Masked). Compute mean $\pm$ std. |
| **Oct 14 – Oct 19** | **Phase 3: Paper Drafting & Results Integration** | Draft Methods, Experiments, and Discussion. Reconcile RQ1-RQ5 with honest claims. Freeze figures. |
| **Oct 20 – Oct 24** | **Phase 4: Implementation Demonstration** | Prepare clean, self-contained inference script and interactive visualization notebook for demo. |
| **Oct 25 – Oct 28** | **Phase 5: Presentation & Viva Preparation** | Prepare slide deck, rehearse defense questions, practice justification of limitations. |
| **Oct 29 – Oct 30** | **Phase 6: Final Verification & Submission** | Clean environment verification, artifact packaging, reproducibility sign-off, project submission. |
