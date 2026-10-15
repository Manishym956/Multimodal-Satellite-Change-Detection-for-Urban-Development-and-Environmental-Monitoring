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
   - Precision, recall, F1, IoU, and false-alarm rate are stored with the confusion counts TP, FP, FN, and TN. For the Scenario 2 Async CNN baseline, FAR recomputed from those counts matches the saved `final_test_far` (0.033602). This note does not re-verify every other row in this pass.
2. **Key Methodological Limitations & Confounds Discovered:**
   - **OSCD SAR Temporal Gap Limitation:** OSCD SAR images are multi-temporal composites without precise acquisition timestamps. Consequently, OSCD cannot evaluate temporal-gap sensitivity (RQ2).
   - **Scenario 1 Val/Test Mismatch:** On OSCD seed 42 only, the Clean Async CNN test F1 is 0.4117 and the validation F1 at the selected epoch is 0.0992 (epoch 26). That test F1 is a single-seed observation. It is not a conclusive or generalizable result. The gap between the validation cities and the test cities is large. This is a description of those cities, not a significance test.
   - **Scenario 2 Masked-Loss Epoch-Progression Confound:** On seed 42, masked-loss training recorded higher all-pixel F1 for Sync CNN (0.1363 vs 0.1102) and for the Cross-Attention Transformer (0.0506 vs 0.0412). Both of those Scenario 2 models used `pos_weight=94.0`. This is not the OSCD loss split, where CNNs used unweighted BCE and the Transformers used `pos_weight=41.0`. The seed-42 checkpoints were selected at later epochs (epoch 30 vs 21 for Sync CNN; epoch 41 vs 23 for the Transformer). Those F1 differences cannot be attributed to masked loss alone. Seeds 43 and 44 are a separate descriptive Sync CNN comparison and are not pooled into this seed-42 contrast.
   - **Async CNN Checkpoint Selection Invariance:** For Async CNN, baseline and masked-loss models are both matched at Epoch 2 (F1: 0.0794 baseline vs 0.0813 masked loss). Masked loss had minimal effect on Async CNN performance. S2-1 Async CNN all-pixel validation F1 peaked at Epoch 18 (0.1600), but those weights were not saved to disk and must not be retrained.
   - **Semantic Scope Limitation:** SpaceNet 7 labels are binary building-detection change proxies (derived from building probability maps thresholded at 0.50). They do not represent agriculture, environmental degradation, or general land-cover shifts. General environmental monitoring claims cannot be made.

---

## 2. Experiment Artifact Inventory

Below is the verified inventory of all completed experiments. Raw records are archived in experiments/research_status/experiment_inventory.csv.

### Table 1: Scenario 1 (OSCD Benchmark, Official 14 Train / 10 Test Cities)
*All models evaluated at $\tau = 0.50$ on pooled test city pixels. Every OSCD row is seed 42. The Clean Async CNN test F1 of 0.4117 is that single seed only and is not a general result. OSCD CNN-family models used unweighted BCE. The two Transformer models used pos_weight=41.0, so CNN and Transformer F1 values in this table are not an architecture-only comparison. Sentinel-1 input on OSCD is VV only.*

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
*Pairing: March 2018 (T1) to December 2019 (T2), one 21-month interval. Scenario 2 SAR input is VV and VH at both dates. Training loss uses pos_weight=94.0. The figure 93.6 is the training-split class ratio (4,718,348 / 50,387) rounded to one decimal place; it is not the weight used in training. Threshold $\tau = 0.50$.*

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

1. **FAR for the Scenario 2 Async CNN baseline:**
   - Saved counts in `experiments/scenario_2/async_dual_stream/metrics.json` are TP 1,601, FP 31,844, FN 5,258, and TN 915,825. FAR = FP / (FP + TN) = 31,844 / (31,844 + 915,825) = 0.033602, which matches `final_test_far`. No saved metrics file for this run contains a test FAR of 0.0348.
2. **Checkpoint selection:**
   - Scenario 2 standard-loss baselines selected the saved checkpoint by all-pixel validation F1.
   - S2-1 masked-loss runs selected the saved checkpoint by valid-pixel validation F1 (`val_f1_valid`).
   - In `experiments/scenario_2_masked_loss/sync_cnn/training_history.csv`, `val_f1_valid` and `val_f1_all` both peak at epoch 30. In `experiments/scenario_2_masked_loss/cross_attention/training_history.csv`, both peak at epoch 41.
   - In `experiments/scenario_2_masked_loss/async_cnn/training_history.csv`, `val_f1_valid` peaks at epoch 2 (0.167661) and `val_f1_all` peaks at epoch 18 (0.159951). The saved selected checkpoint is epoch 2 (`validation_best_epoch` in `experiments/scenario_2_masked_loss/async_cnn/metrics.json`). Epoch 18 is not that checkpoint. The baseline Async CNN history in `experiments/scenario_2/async_dual_stream/training_history.csv` has one validation F1 series, and that series peaks at epoch 2 (0.183204), matching `validation_best_f1`.
3. **Epoch Progression Confound in S2-1 Gains:**
   - The reported improvement for Sync CNN (+0.0261 F1) and Cross-Attention (+0.0094 F1) cannot be disentangled from the fact that they trained 9 and 18 epochs longer than their respective baselines before peaking.

---

## 4. Reconciliation of RRD Research Questions

| RQ | Original Description | Current Status | Supporting Evidence & Honest Classification |
|:---:|:---|:---:|:---|
| **RQ1** | Can historical Sentinel-2 optical imagery improve Sentinel-1 change detection when synchronous optical is unavailable? | **Partially investigated** | **Single-seed OSCD observation (seed 42), not conclusive or generalizable:** Clean Async CNN test F1 = 0.4117, SAR-only = 0.3558 (difference +0.0559), and Naive Async = 0.2052. Those three CNNs used unweighted BCE. The Cross-Attention Transformer test F1 is 0.3564 and used pos_weight=41.0, so it is not on the same loss as the CNNs.<br>**Not demonstrated on the Scenario 2 21-month building-proxy split:** Async CNN test F1 = 0.0794 vs Sync CNN 0.1102. Both used pos_weight=94.0. |
| **RQ2** | How does temporal gap between historical optical imagery and target SAR affect performance? | **Missing** | OSCD SAR images are multi-temporal composites without precise acquisition timestamps, so those runs do not measure gap sensitivity. Scenario 2 training used one fixed pair (March 2018 to December 2019, about 21 months). A single interval does not establish temporal-gap sensitivity. RQ2 is not complete. |
| **RQ3** | Can additional SAR information (VV and VH polarization) improve change discrimination? | **Missing** | OSCD experiments use Sentinel-1 VV only. Scenario 2 stacks VV and VH (VV and VH at T1 and at T2) in the trained models. No run compares VV only, VH only, and VV+VH, so the polarization question is unanswered. RQ3 is not complete. |
| **RQ4** | How does the proposed asynchronous multimodal approach compare with unimodal, conventional fusion, and dual-stream baselines? | **Partial** | OSCD seed 42 records SAR-only (0.3558), optical-only (0.2663), early fusion (0.3520), sync dual-stream (0.3868), naive async (0.2052), and clean async (0.4117). The clean-async value is that single seed only. CNN-family OSCD models used unweighted BCE; the two Transformers used pos_weight=41.0, so that order is not an architecture-only comparison. Scenario 2 records Sync CNN (0.1102), Async CNN (0.0794), and Cross-Attention (0.0412), all with pos_weight=94.0, and has no SAR-only or optical-only baseline. |
| **RQ5** | Under what temporal and environmental conditions does historical optical guidance provide greatest benefit? | **Partial; environmental analysis blocked** | Error maps and qualitative inspections exist for urban scenes, including rapid construction and unmasked cloud or sensor-boundary regions (L15-1289E). They do not identify agricultural or environmental change. Scenario 2 labels are building-probability proxies. Environmental pseudo-change analysis is not complete. |

---

## 5. Matched Sync CNN Multi-Seed Runs (Seeds 43 and 44)

The matched Sync CNN runs for seeds 43 and 44 are on disk under `experiments/scenario_2_multiseed/`. They compare standard weighted BCE with masked weighted BCE. Two matched seeds are a descriptive comparison only. They do not establish statistical significance, and they do not complete RQ2, RQ3, or environmental pseudo-change analysis.

### Experimental Setup & Controls
- **Model:** Sync Dual-Stream CNN (`DualStreamUNet`, 27,281,793 parameters in the multiseed metrics).
- **Split:** Locked Strategy B (20 train, 4 val, 4 test sites from split_definition.json).
- **Input Channels:** SAR T1+T2 is 4 channels (VV and VH at each date). Optical T1+T2 is 20 channels (10 Sentinel-2 bands at each date). Total input is 24 channels. OSCD remains VV only; this 4-channel SAR stack is Scenario 2 only.
- **Hyperparameters:** Batch size = 64, Crop size = 32, Crops/site = 64, LR = 1e-4 (Adam), Epochs = 50, training pos_weight = 94.0. The class ratio rounded to one decimal place is 93.6; that ratio is not the training weight.
- **Experimental Variable:** Standard weighted BCE on all pixels vs. masked weighted BCE on valid SAR pixels. Both use training pos_weight=94.0.
- **Checkpoint Selection Rule:** Seeds 43 and 44 selected both loss conditions by the highest valid-pixel validation F1. Seed 42 standard BCE used all-pixel validation F1 and is not part of that matched pair.
- **Evaluation:** Dual-protocol testing at $\tau = 0.50$ and calibrated $\tau^*$.

### Runtime and Resource Estimates
- Each Sync CNN 50-epoch run takes **~513 to 557 seconds** (~8.5 to 9.3 minutes) on the NVIDIA RTX 4060 Laptop GPU.
- The four runs are the Seed 43 and Seed 44 standard and masked Sync CNN checkpoints under `experiments/scenario_2_multiseed/`.
- Recorded training time for those runs is about 8.5 to 9.3 minutes each.
- The paired all-pixel F1 differences are descriptive. They are reported in `docs/SCENARIO_2_MULTISEED_SYNC_REPORT.md` and `docs/paper/02_results_v2.md`.

---

## 6. Recommended Task Order Through October 30

| Date Window | Priority / Task | Key Deliverables & Milestones |
|---|---|---|
| **Oct 11 (Today)** | **Phase 1: Freeze & Audit** | Establish artifact inventory, compute hashes, lock baseline records, freeze code. |
| **Oct 12 – Oct 13** | **Phase 2: Multi-Seed Experiment (Optional/Controlled)** | Four Sync CNN runs (seeds 43 and 44; standard weighted BCE vs masked weighted BCE) are stored under `experiments/scenario_2_multiseed/`. Report the paired differences as descriptive results from two matched seeds. |
| **Oct 14 – Oct 19** | **Phase 3: Paper Drafting & Results Integration** | Draft Methods, Experiments, and Discussion. Reconcile RQ1-RQ5 with honest claims. Freeze figures. |
| **Oct 20 – Oct 24** | **Phase 4: Implementation Demonstration** | Prepare clean, self-contained inference script and interactive visualization notebook for demo. |
| **Oct 25 – Oct 28** | **Phase 5: Presentation & Viva Preparation** | Prepare slide deck, rehearse defense questions, practice justification of limitations. |
| **Oct 29 – Oct 30** | **Phase 6: Final Verification & Submission** | Clean environment verification, artifact packaging, reproducibility sign-off, project submission. |
