# Scenario 2 Checkpoint-Selection Ablation

**Audit Date:** 2026-10-10  
**Status:** Complete (Read-Only Independent Evaluation)  
**Execution Environment:** NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB VRAM), PyTorch 2.5.1+cu124, CUDA 12.4  
**Primary Objective:** Isolate the impact of the checkpoint-selection protocol on apparent masked-loss gains, distinguishing genuine architectural/loss effects from validation-selection and epoch progression confounds.

---

## 1. Executive Summary

During the post-experiment audit of Experiment S2-1 (Masked-Loss Training), a procedural difference was detected: baseline checkpoints were selected by **All-Pixel Validation F1** (`val_f1`), whereas S2-1 checkpoints were selected by **Valid-Pixel Validation F1** (`val_f1_valid`).

This ablation investigated whether this difference altered checkpoint selection and whether the reported test performance shifts are attributable to masked loss or to the selection criterion:

1. **Within-S2-1 Checkpoint Selection Agreement:**
   - **Sync CNN:** Both all-pixel and valid-pixel validation F1 peaked at **the exact same epoch (Epoch 30)** (`val_f1_all = 0.2789`, `val_f1_valid = 0.2878`). The selection rule caused **zero difference** in the saved checkpoint.
   - **Cross-Attention Transformer:** Both all-pixel and valid-pixel validation F1 peaked at **the exact same epoch (Epoch 41)** (`val_f1_all = 0.2088`, `val_f1_valid = 0.2223`). The selection rule caused **zero difference** in the saved checkpoint.
   - **Async CNN:** Valid-pixel validation F1 peaked at **Epoch 2** (`val_f1_valid = 0.1677`), whereas all-pixel validation F1 peaked at **Epoch 18** (`val_f1_all = 0.1600`).
2. **Checkpoint Availability & Limitation:**
   - Checkpoints saved on disk during training are strictly `best.pt` and `last.pt` (Epoch 50).
   - Weights for S2-1 Async CNN Epoch 18 were not saved to disk. In accordance with non-retraining constraints, Epoch 18 cannot be evaluated.
3. **Controlled vs. Confounded Comparisons:**
   - **Async CNN:** Both Baseline and S2-1 evaluated checkpoints are **Epoch 2**. This comparison is **epoch-matched**; however, it remains **partially confounded** by the fact that the two runs used different validation metrics during training monitoring and had different early loss trajectories. Across identical test sets at $\tau = 0.50$, Async CNN shows minimal change: Baseline F1 = **0.0794** vs. S2-1 F1 = **0.0813** ($\Delta = +0.0019$).
   - **Sync CNN & Transformer:** Within S2-1, the selection rule is invariant. However, relative to baseline, the models selected **substantially later epochs** (Sync CNN: Epoch 21 $\to$ 30; Transformer: Epoch 23 $\to$ 41). Consequently, comparing S2-1 to Baseline is **confounded by training epoch progression**, not isolated to masked loss alone.

---

## 2. Training History Analysis: Criterion Comparison

Evaluation of `training_history.csv` across all 50 epochs revealed the following peaks on the 4 validation sites:

| Model Family | Metric | Best Epoch | Peak Value | Corresponding Alternate Metric |
|---|---|---|---|---|
| **S2-1 Async CNN** | `val_f1_valid` (Valid-Pixel) | **Epoch 2** | **0.1677** | 0.1528 (`val_f1_all`) |
| | `val_f1_all` (All-Pixel) | **Epoch 18** | **0.1600** | 0.1624 (`val_f1_valid`) |
| **S2-1 Sync CNN** | `val_f1_valid` (Valid-Pixel) | **Epoch 30** | **0.2878** | 0.2789 (`val_f1_all`) |
| | `val_f1_all` (All-Pixel) | **Epoch 30** | **0.2789** | 0.2878 (`val_f1_valid`) |
| **S2-1 Cross-Attention** | `val_f1_valid` (Valid-Pixel) | **Epoch 41** | **0.2223** | 0.2088 (`val_f1_all`) |
| | `val_f1_all` (All-Pixel) | **Epoch 41** | **0.2088** | 0.2223 (`val_f1_valid`) |

### Top 3 Epoch Trajectories

- **S2-1 Sync CNN:**
  - Rank 1: Epoch 30 (`val_f1_all = 0.2789`, `val_f1_valid = 0.2878`)
  - Rank 2: Epoch 41 (`val_f1_all = 0.2743`, `val_f1_valid = 0.2824`)
  - Rank 3: Epoch 36 (`val_f1_all = 0.2689`, `val_f1_valid = 0.2772`)
  *(Strictly monotonic ranking across both criteria).*
- **S2-1 Cross-Attention:**
  - Rank 1: Epoch 41 (`val_f1_all = 0.2088`, `val_f1_valid = 0.2223`)
  - Rank 2: Epoch 42 (`val_f1_all = 0.1733`, `val_f1_valid = 0.1799`)
  - Rank 3: Epoch 40 (`val_f1_all = 0.1547`, `val_f1_valid = 0.1606`)
  *(Strictly monotonic ranking across both criteria).*
- **S2-1 Async CNN:**
  - Rank 1 by valid-pixel: Epoch 2 (`val_f1_valid = 0.1677`, `val_f1_all = 0.1528`)
  - Rank 2 by valid-pixel: Epoch 18 (`val_f1_valid = 0.1624`, `val_f1_all = 0.1600`)
  - Rank 1 by all-pixel: Epoch 18 (`val_f1_all = 0.1600`)

---

## 3. Checkpoint Availability & Verification

The filesystem on disk preserves the following weights under `checkpoints/`:

```
checkpoints/scenario_2/
  ├── async_dual_stream/best.pt      (Epoch 2, Baseline)
  ├── sync_dual_stream/best.pt       (Epoch 21, Baseline)
  └── cross_attention/best.pt        (Epoch 23, Baseline)
checkpoints/scenario_2_masked_loss/
  ├── async_cnn/best.pt              (Epoch 2, S2-1)
  ├── sync_cnn/best.pt               (Epoch 30, S2-1)
  └── cross_attention/best.pt        (Epoch 41, S2-1)
```

- **Async CNN Epoch 18:** Not saved. Per guidelines, no models were retrained.
- **Sync CNN & Cross-Attention:** The available `best.pt` represents the global maximum for **both** all-pixel and valid-pixel validation F1.

---

## 4. Primary Test Evaluation ($\tau = 0.50$)

All checkpoints were independently evaluated across the four locked test sites (`L15-0924E`, `L15-1289E`, `L15-1298E`, `L15-1438E`) using full-tile inference:

### 4.1 All-Pixel Protocol

| Model Family | Run / Checkpoint | Eval Epoch | TP | FP | FN | TN | Precision | Recall | **F1** | IoU | FAR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Async CNN** | Baseline | 2 | 1,601 | 31,844 | 5,258 | 915,825 | 0.0479 | 0.2334 | **0.0794** | 0.0414 | 0.0336 |
| | S2-1 Masked (Saved) | 2 | 1,994 | 40,187 | 4,865 | 907,482 | 0.0473 | 0.2907 | **0.0813** | 0.0424 | 0.0424 |
| | S2-1 Masked (All-Pixel Rule) | 18 | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A (Weights not saved)* | *N/A* | *N/A* |
| **Sync CNN** | Baseline | 21 | 3,896 | 59,965 | 2,963 | 887,704 | 0.0610 | 0.5680 | **0.1102** | 0.0583 | 0.0633 |
| | S2-1 Masked (Both Rules) | 30 | 3,664 | 43,241 | 3,195 | 904,428 | 0.0781 | 0.5342 | **0.1363** | 0.0731 | 0.0456 |
| **Cross-Attention** | Baseline | 23 | 3,913 | 179,292 | 2,946 | 768,377 | 0.0214 | 0.5705 | **0.0412** | 0.0210 | 0.1892 |
| | S2-1 Masked (Both Rules) | 41 | 3,182 | 115,791 | 3,677 | 831,878 | 0.0267 | 0.4639 | **0.0506** | 0.0259 | 0.1222 |

### 4.2 Valid-Pixel Protocol

| Model Family | Run / Checkpoint | Eval Epoch | vTP | vFP | vFN | vTN | vPrecision | vRecall | **vF1** | vIoU | vFAR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Async CNN** | Baseline | 2 | 1,580 | 30,893 | 5,236 | 787,559 | 0.0487 | 0.2318 | **0.0804** | 0.0419 | 0.0377 |
| | S2-1 Masked (Saved) | 2 | 1,973 | 39,239 | 4,843 | 779,213 | 0.0479 | 0.2895 | **0.0822** | 0.0428 | 0.0479 |
| **Sync CNN** | Baseline | 21 | 3,871 | 59,226 | 2,945 | 759,226 | 0.0613 | 0.5679 | **0.1107** | 0.0586 | 0.0724 |
| | S2-1 Masked (Both Rules) | 30 | 3,639 | 42,438 | 3,177 | 776,014 | 0.0790 | 0.5339 | **0.1376** | 0.0739 | 0.0519 |
| **Cross-Attention** | Baseline | 23 | 3,886 | 178,215 | 2,930 | 640,237 | 0.0213 | 0.5701 | **0.0411** | 0.0210 | 0.2177 |
| | S2-1 Masked (Both Rules) | 41 | 3,156 | 114,720 | 3,660 | 703,732 | 0.0268 | 0.4630 | **0.0506** | 0.0260 | 0.1402 |

---

## 5. Per-Site Breakdown ($\tau = 0.50$, All-Pixel)

| Model & Checkpoint | Site ID | Total Pixels | Change Pixels | TP | FP | FN | TN | Precision | Recall | **F1** | FAR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Baseline Async CNN** (Ep 2) | `L15-0924E` | 238,632 | 723 | 171 | 3,523 | 552 | 234,386 | 0.0463 | 0.2365 | **0.0774** | 0.0148 |
| | `L15-1289E` | 238,632 | 3,195 | 866 | 24,848 | 2,329 | 210,589 | 0.0337 | 0.2710 | **0.0599** | 0.1055 |
| | `L15-1298E` | 238,632 | 1,339 | 217 | 1,622 | 1,122 | 235,671 | 0.1180 | 0.1621 | **0.1366** | 0.0068 |
| | `L15-1438E` | 238,632 | 1,602 | 347 | 1,851 | 1,255 | 235,179 | 0.1579 | 0.2166 | **0.1826** | 0.0078 |
| **S2-1 Async CNN** (Ep 2) | `L15-0924E` | 238,632 | 723 | 181 | 3,294 | 542 | 234,615 | 0.0521 | 0.2503 | **0.0862** | 0.0138 |
| | `L15-1289E` | 238,632 | 3,195 | 1,183 | 33,328 | 2,012 | 202,109 | 0.0343 | 0.3703 | **0.0627** | 0.1416 |
| | `L15-1298E` | 238,632 | 1,339 | 266 | 1,738 | 1,073 | 235,555 | 0.1327 | 0.1987 | **0.1591** | 0.0073 |
| | `L15-1438E` | 238,632 | 1,602 | 364 | 1,827 | 1,238 | 235,203 | 0.1661 | 0.2272 | **0.1919** | 0.0077 |
| **Baseline Sync CNN** (Ep 21) | `L15-0924E` | 238,632 | 723 | 352 | 10,355 | 371 | 227,554 | 0.0329 | 0.4869 | **0.0616** | 0.0435 |
| | `L15-1289E` | 238,632 | 3,195 | 2,484 | 41,876 | 711 | 193,561 | 0.0560 | 0.7775 | **0.1045** | 0.1779 |
| | `L15-1298E` | 238,632 | 1,339 | 299 | 2,609 | 1,040 | 234,684 | 0.1028 | 0.2233 | **0.1408** | 0.0110 |
| | `L15-1438E` | 238,632 | 1,602 | 761 | 5,125 | 841 | 231,905 | 0.1293 | 0.4750 | **0.2033** | 0.0216 |
| **S2-1 Sync CNN** (Ep 30) | `L15-0924E` | 238,632 | 723 | 358 | 11,406 | 365 | 226,503 | 0.0304 | 0.4952 | **0.0573** | 0.0479 |
| | `L15-1289E` | 238,632 | 3,195 | 2,355 | 25,776 | 840 | 209,661 | 0.0837 | 0.7371 | **0.1504** | 0.1095 |
| | `L15-1298E` | 238,632 | 1,339 | 252 | 1,564 | 1,087 | 235,729 | 0.1388 | 0.1882 | **0.1597** | 0.0066 |
| | `L15-1438E` | 238,632 | 1,602 | 699 | 4,495 | 903 | 232,535 | 0.1346 | 0.4363 | **0.2057** | 0.0190 |
| **Baseline Cross-Attn** (Ep 23) | `L15-0924E` | 238,632 | 723 | 166 | 3,113 | 557 | 234,796 | 0.0506 | 0.2296 | **0.0830** | 0.0131 |
| | `L15-1289E` | 238,632 | 3,195 | 2,769 | 166,162 | 426 | 69,275 | 0.0164 | 0.8667 | **0.0322** | 0.7058 |
| | `L15-1298E` | 238,632 | 1,339 | 329 | 2,710 | 1,010 | 234,583 | 0.1083 | 0.2457 | **0.1503** | 0.0114 |
| | `L15-1438E` | 238,632 | 1,602 | 649 | 7,307 | 953 | 232,723 | 0.0816 | 0.4051 | **0.1358** | 0.0308 |
| **S2-1 Cross-Attn** (Ep 41) | `L15-0924E` | 238,632 | 723 | 158 | 3,107 | 565 | 234,802 | 0.0484 | 0.2185 | **0.0792** | 0.0131 |
| | `L15-1289E` | 238,632 | 3,195 | 2,372 | 108,311 | 823 | 127,126 | 0.0214 | 0.7424 | **0.0417** | 0.4600 |
| | `L15-1298E` | 238,632 | 1,339 | 210 | 1,492 | 1,129 | 235,801 | 0.1234 | 0.1568 | **0.1381** | 0.0063 |
| | `L15-1438E` | 238,632 | 1,602 | 442 | 2,881 | 1,160 | 234,149 | 0.1330 | 0.2759 | **0.1795** | 0.0122 |

---

## 6. Secondary Analysis: Validation-Calibrated Thresholds ($\tau^*$)

Optimal thresholds $\tau^*$ were derived strictly on the 4 validation sites without referencing test set labels:

| Model Family | Variant | Eval Epoch | $\tau^*$ | All-Pixel Prec | All-Pixel Rec | **All-Pixel F1** | All-Pixel FAR | Valid-Pixel F1 | Valid-Pixel FAR |
|---|---|---|---|---|---|---|---|---|---|
| **Async CNN** | Baseline | 2 | 0.520 | 0.0491 | 0.2271 | **0.0807** | 0.0318 | 0.0818 | 0.0357 |
| | S2-1 Masked | 2 | 0.520 | 0.0486 | 0.2751 | **0.0827** | 0.0390 | 0.0835 | 0.0440 |
| **Sync CNN** | Baseline | 21 | 0.830 | 0.1313 | 0.4139 | **0.1994** | 0.0198 | 0.2009 | 0.0225 |
| | S2-1 Masked | 30 | 0.780 | 0.1346 | 0.3283 | **0.1909** | 0.0153 | 0.1934 | 0.0171 |
| **Cross-Attention** | Baseline | 23 | 0.730 | 0.0261 | 0.3872 | **0.0488** | 0.1048 | 0.0487 | 0.1204 |
| | S2-1 Masked | 41 | 0.580 | 0.0289 | 0.4044 | **0.0540** | 0.0982 | 0.0541 | 0.1126 |

---

## 7. Synthesis: Controlled vs. Confounded Findings

### 7.1 Within S2-1 Checkpoint-Selection Rule Comparison
- **Conclusion:** For **Sync CNN** and **Cross-Attention**, the selection criterion (all-pixel vs. valid-pixel validation F1) had **zero influence** on checkpoint selection. Both models peaked at the identical epochs (Epoch 30 and Epoch 41, respectively).
- **Limitation:** For **Async CNN**, the all-pixel best epoch was Epoch 18 (`val_f1_all = 0.1600`), but weights for Epoch 18 are not available on disk.

### 7.2 Comparison against Baseline (Controlled vs. Confounded Status)

| Model Family | Evaluation Epochs | Status of Comparison | Scientific Assessment |
|---|---|---|---|
| **Async CNN** | Baseline: Ep 2<br>S2-1: Ep 2 | **Epoch-matched, but partially confounded** | Both models were evaluated at Epoch 2. However, training loss formulations differed, and Async CNN early-epoch training dynamics under extreme class weighting (`pos_weight = 94.0`) exhibit high variance. Test F1 changed by only +0.0019 (0.0794 $\to$ 0.0813), with FAR increasing from 0.0336 to 0.0424. This change is within noise variance. |
| **Sync CNN** | Baseline: Ep 21<br>S2-1: Ep 30 | **Confounded by training epoch** | Within S2-1, Epoch 30 is the true peak under both validation criteria. However, because Epoch 30 is 9 epochs later than the baseline's Epoch 21, the observed test gain at $\tau=0.50$ (0.1102 $\to$ 0.1363, with FP dropping from 59,965 to 43,241) cannot be cleanly attributed to masked loss alone versus additional convergence / epoch progression. Interestingly, under calibrated thresholds, Baseline ($\tau^*=0.83$) achieves F1 = **0.1994**, exceeding S2-1 ($\tau^*=0.78$) at F1 = **0.1909**. |
| **Cross-Attention** | Baseline: Ep 23<br>S2-1: Ep 41 | **Confounded by training epoch** | Within S2-1, Epoch 41 is the peak under both validation criteria. However, evaluating at Epoch 41 represents 18 additional epochs of training relative to baseline Epoch 23. While FP on `L15-1289E` dropped by 57,851 pixels, recall also dropped from 86.7% to 74.2%. The probability collapse is damped but not resolved, and the improvement cannot be attributed exclusively to masked loss. |

---

## 8. Final Audit Sign-Off

- **Artifact Integrity:** All baseline checkpoints, metric records, split definitions (`split_definition.json`), and S2-1 experiment directories remain completely intact and unaltered.
- **Reproducibility:** Re-evaluated metrics are identical to those generated in previous audit passes.
- **Reporting Recommendation:** When documenting Scenario 2 in research manuscripts or reports:
  1. Clearly state that within S2-1, Sync CNN and Cross-Attention checkpoint selection is invariant to the choice between all-pixel and valid-pixel validation F1.
  2. Disclose that comparisons between S2-1 and baseline for Sync CNN and Cross-Attention are confounded by epoch progression (Epochs 30/41 vs. 21/23).
  3. Avoid claiming that masked loss definitively caused the +23.7% Sync CNN gain at $\tau=0.50$, as threshold calibration alone recovers F1 = 0.1994 on the baseline checkpoint.
