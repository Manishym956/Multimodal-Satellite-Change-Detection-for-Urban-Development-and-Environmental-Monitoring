# Scenario 2 Missing-Data, Error & Improvement Audit

**Audit Date:** 2026-10-10  
**Status:** Completed (Inspection-Only Phase — No Checkpoints or Baselines Overwritten)  
**Execution Environment:** NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB VRAM), PyTorch 2.5.1+cu124, CUDA 12.4  

---

## 1. Executive Summary

This audit completes the four required diagnostic phases (Phase A through Phase D) for Scenario 2 (SEN12-MT / SpaceNet 7 derived building-footprint change proxy, 21-month baseline pair: March 2018 to December 2019):
1. **Validity Mask Audit (Phase A):** Quantified raw NaN and valid pixel distributions across all 28 sites. Optical channels are 100% valid across all sites (0.000% NaN). Several training sites and test site `L15-0924E` contain substantial SAR NaNs (up to 78% invalid). Crucially, the problematic test site `L15-1289E` is **99.76% valid** in SAR; missing data is **not** the cause of the Transformer false-alarm surge.
2. **Error Overlap & Visualization (Phase B):** Rendered 12-panel multi-modal diagnostic panels and error overlays for `L15-1289E`. **99.72% of the Transformer's 166,162 false positives on this site occur on completely valid SAR and Optical pixels.** The failure is a probability calibration collapse (mean predicted probability ~0.60–0.70 across the whole tile), triggering change predictions across 70.8% of the site even where SAR backscatter differences are near zero.
3. **Threshold Calibration (Phase C):** Swept decision thresholds strictly on the 4 validation sites. Because models were trained with high class imbalance weighting (`pos_weight = 94.0`), predicted probabilities are shifted upwards. Selecting $\tau^*$ strictly on validation data:
   - **Sync Dual-Stream CNN:** $\tau^* = 0.830$ boosts test Precision from **0.0610 to 0.1313**, cuts test False Alarm Rate from **0.0633 to 0.0198** (>3× reduction), and almost doubles test F1 from **0.1102 to 0.1994** (IoU from 0.0583 to 0.1107).
   - **Async Dual-Stream CNN:** $\tau^* = 0.520$ yields Test F1 = **0.0807** (vs 0.0794 at 0.50).
   - **Cross-Attention Transformer:** $\tau^* = 0.730$ cuts test False Alarm Rate from **0.1892 to 0.1048** (almost halved).
4. **Training Loss Status:** Verified that the current training pipeline (`train_scenario2_*.py`) computes `BCEWithLogitsLoss` over **all crop pixels unconditionally**, forcing models to compute loss on dummy `0.0` values where SAR was originally missing.
5. **Controlled Roadmap (Phase D):** Formulated an incremental, single-variable experimental roadmap prioritizing masked loss and calibrated thresholds.

---

## 2. Phase A: Validity Mask Audit Across All 28 Sites

### 2.1 Optical Validity
- **Sentinel-2 $T_1$ (10 bands):** **0.000% NaN** across all 28 sites.
- **Sentinel-2 $T_2$ (10 bands, sync only):** **0.000% NaN** across all 28 sites.
- **Conclusion:** Optical data quality is clean and complete; no invalid optical pixels exist in the locked 28-site split.

### 2.2 SAR Validity ($T_1$ and $T_2$ Combined)
A pixel is defined as **valid** if and only if both VV and VH channels are finite (non-NaN) at both $T_1$ (March 2018) and $T_2$ (December 2019):
$$\text{Valid}_{\text{SAR}} = \text{Valid}_{\text{S1\_T1}} \land \text{Valid}_{\text{S1\_T2}}$$

#### Training Set (20 Sites, 4,768,735 Pixels Total)
- **High-NaN outlier sites in training:**
  - `L15-0434E-1218N_1736_3318_13`: S1 $T_1$ invalid = 63.84%, S1 $T_2$ invalid = 52.37% (**Combined Valid: 21.70%**; 78.30% invalid!)
  - `L15-1335E-1166N_5342_3524_13`: S1 $T_1$ invalid = 48.02%, S1 $T_2$ invalid = 50.60% (**Combined Valid: 29.26%**; 70.74% invalid!)
  - `L15-1296E-1198N_5184_3399_13`: S1 $T_1$ invalid = 37.33%, S1 $T_2$ invalid = 32.29% (**Combined Valid: 47.92%**; 52.08% invalid!)
  - `L15-1389E-1284N_5557_3054_13`: S1 $T_1$ invalid = 20.81%, S1 $T_2$ invalid = 34.43% (**Combined Valid: 59.61%**)
  - `L15-1209E-1113N_4838_3737_13`: S1 $T_1$ invalid = 20.70%, S1 $T_2$ invalid = 24.54% (**Combined Valid: 68.94%**)
  - `L15-1276E-1107N_5105_3761_13`: S1 $T_1$ invalid = 5.14%, S1 $T_2$ invalid = 21.60% (**Combined Valid: 75.48%**)
- **Summary for Training:** 14 of 20 training sites have $>0.1\%$ invalid pixels. 4 training sites have $>50\%$ missing SAR data.

#### Validation Set (4 Sites, 954,528 Pixels Total)
- `L15-0977E-1187N_3911_3441_13`: S1 $T_1$ invalid = 36.70%, S1 $T_2$ invalid = 33.34% (**Combined Valid: 52.81%**)
- `L15-0487E-1246N_1950_3207_13`: S1 $T_1$ invalid = 1.19%, S1 $T_2$ invalid = 8.73% (**Combined Valid: 90.74%**)
- `L15-0457E-1135N_1831_3648_13`: Combined Valid: 99.93%
- `L15-1716E-1211N_6864_3345_13`: Combined Valid: 99.87%

#### Test Set (4 Sites, 954,528 Pixels Total)
- `L15-0924E-1108N_3699_3757_13`: S1 $T_1$ invalid = 28.32%, S1 $T_2$ invalid = 36.57% (**Combined Valid: 55.40%**; 106,435 invalid pixels)
- `L15-1298E-1322N_5193_2903_13`: S1 $T_1$ invalid = 3.10%, S1 $T_2$ invalid = 5.76% (**Combined Valid: 92.28%**; 18,431 invalid pixels)
- `L15-1438E-1134N_5753_3655_13`: S1 $T_1$ invalid = 0.76%, S1 $T_2$ invalid = 1.25% (**Combined Valid: 98.40%**; 3,827 invalid pixels)
- `L15-1289E-1169N_5156_3514_13`: S1 $T_1$ invalid = 0.24%, S1 $T_2$ invalid = 0.002% (**Combined Valid: 99.76%**; **only 567 invalid pixels**)

---

### 2.3 Evaluation on Valid Pixels vs All Pixels

Evaluating the three trained checkpoints on valid pixels only ($\text{Valid}_{\text{SAR}}$) versus all pixels:

| Model | Evaluation Subset | Precision | Recall | F1 | IoU | FAR | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|---|
| **Async Dual-Stream CNN** | All pixels | 0.0479 | 0.2334 | **0.0794** | 0.0414 | 0.0336 | 1,601 | 31,844 | 5,258 | 915,825 |
| | **Valid pixels only** | 0.0487 | 0.2318 | **0.0804** | 0.0419 | 0.0377 | 1,580 | 30,893 | 5,236 | 787,559 |
| **Sync Dual-Stream CNN** | All pixels | 0.0610 | 0.5680 | **0.1102** | 0.0583 | 0.0633 | 3,896 | 59,965 | 2,963 | 887,704 |
| | **Valid pixels only** | 0.0613 | 0.5679 | **0.1107** | 0.0586 | 0.0724 | 3,871 | 59,226 | 2,945 | 759,226 |
| **Cross-Attention Transformer** | All pixels | 0.0214 | 0.5705 | **0.0412** | 0.0210 | 0.1892 | 3,913 | 179,292 | 2,946 | 768,377 |
| | **Valid pixels only** | 0.0213 | 0.5701 | **0.0411** | 0.0210 | 0.2177 | 3,886 | 178,215 | 2,930 | 640,237 |

#### Key Takeaway from Valid-Pixel Evaluation
Filtering to valid pixels slightly cleans up metrics on site `L15-0924E` (e.g. Async CNN F1 on that site increases from 0.0774 to 0.0874; Transformer F1 increases from 0.0830 to 0.0918). However, pooled test F1 is virtually unchanged across all models because the dominant error source is false alarms on `L15-1289E`, which is 99.76% valid.

---

## 3. Phase B: Visualization and Error Analysis on Site `L15-1289E`

**Artifacts Generated:**
- Multi-panel Diagnostic Panel: [`experiments/scenario_2/error_analysis/L15_1289E_visual_panel.png`](file:///C:/Users/Manishym/Researchpapers/satellite%E2%80%91change%E2%80%91detection/projects/multimodal-satellite-change-detection/experiments/scenario_2/error_analysis/L15_1289E_visual_panel.png)
- Dedicated Error Maps: `async_cnn_error_map.png`, `sync_cnn_error_map.png`, `cross_attn_error_map.png`
- Overlap Statistics: [`experiments/scenario_2/error_analysis/error_overlap_stats.json`](file:///C:/Users/Manishym/Researchpapers/satellite%E2%80%91change%E2%80%91detection/projects/multimodal-satellite-change-detection/experiments/scenario_2/error_analysis/error_overlap_stats.json)

### 3.1 Overlap Statistics on Site `L15-1289E`
- **Total Site Pixels:** 238,632
- **Ground Truth Building Change:** 3,195 pixels (1.339%)
- **Total Invalid SAR Pixels:** **567 pixels (0.238%)**

| Metric | Async CNN | Sync CNN | Cross-Attention Transformer |
|---|---|---|---|
| True Positives (TP) | 866 | 2,484 | 2,769 |
| False Positives (FP) | 24,848 | 41,876 | **166,162** |
| False Negatives (FN) | 2,329 | 711 | 426 |
| True Negatives (TN) | 210,589 | 193,561 | 69,275 |
| FP on Invalid Pixels | 259 | 258 | 459 |
| **% of Model's FPs on Invalid Pixels** | **1.04%** | **0.62%** | **0.28%** |
| % of Invalid Pixels Predicted FP | 45.68% | 45.50% | 80.95% |
| Mean $\|VV_{T2} - VV_{T1}\|$ on True Positives | 0.2728 | 0.1847 | 0.1799 |
| Mean $\|VV_{T2} - VV_{T1}\|$ on False Positives | 0.1728 | 0.1420 | **0.0945** |
| Mean $\|VV_{T2} - VV_{T1}\|$ on True Negatives | 0.0741 | 0.0720 | 0.0604 |

### 3.2 Visual Diagnosis & Physical Mechanism
1. **False Positives are Not Caused by Sensor NaNs:** 99.72% of the Transformer's false alarms occur on genuine, valid SAR pixels.
2. **Probability Shift in Cross-Attention:** The Transformer predicts positive change across **70.8% of the entire site area** (168,931 pixels). Its mean predicted probability across this site is ~0.60–0.70.
3. **Hypothesis for Cross-Attention Failure:** 
   In `TransformerCrossAttention`, SAR tokens serve as Query, and optical tokens serve as Key/Value. When historical optical context ($T_1$) contains widespread urban textures with dense high-frequency building edges, but contemporary optical context is missing, the cross-attention mechanism strongly attends to historical building structures. When coupled with the high training positive weight (`pos_weight = 94.0`), the model defaults to an extreme over-prediction mode whenever historical buildings are present, even when SAR temporal difference is near baseline ($\Delta \text{SAR} = 0.095$).

---

## 4. Phase C: Validation-Only Threshold Calibration

**Artifacts Generated:**
- PR & Threshold Curves: [`experiments/scenario_2/threshold_calibration/validation_pr_curves.png`](file:///C:/Users/Manishym/Researchpapers/satellite%E2%80%91change%E2%80%91detection/projects/multimodal-satellite-change-detection/experiments/scenario_2/threshold_calibration/validation_pr_curves.png)
- Detailed Calibration Report: [`experiments/scenario_2/threshold_calibration/threshold_calibration_results.json`](file:///C:/Users/Manishym/Researchpapers/satellite%E2%80%91change%E2%80%91detection/projects/multimodal-satellite-change-detection/experiments/scenario_2/threshold_calibration/threshold_calibration_results.json)

### 4.1 Calibration Methodology
- Decision threshold $\tau$ was swept over $[0.01, 0.99]$ **strictly on the 4 validation sites**.
- Optimal threshold $\tau^*$ was chosen by maximizing Validation F1:
  $$\tau^* = \arg\max_\tau \text{F1}_{\text{val}}(\tau)$$
- **Test labels were never accessed during threshold selection.**
- The frozen $\tau^*$ was then evaluated on the official 4 test sites and compared against default $\tau = 0.50$.

### 4.2 Calibration Results

| Model | Val $\tau^*$ | Val F1 ($\tau = 0.5$) | Val F1 ($\tau^*$) | Test F1 ($\tau = 0.5$) | **Test F1 ($\tau^*$)** | Test Prec ($\tau^*$) | Test Rec ($\tau^*$) | Test FAR ($\tau^*$) |
|---|---|---|---|---|---|---|---|---|
| **Async Dual-Stream CNN** | **0.520** | 0.1832 | **0.1834** | 0.0794 | **0.0807** (+1.6%) | 0.0491 | 0.2271 | 0.0318 |
| **Sync Dual-Stream CNN** | **0.830** | 0.2663 | **0.3501** | 0.1102 | **0.1994** (+80.9%) | **0.1313** | 0.4139 | **0.0198** |
| **Cross-Attention Transformer** | **0.730** | 0.1328 | **0.1657** | 0.0412 | **0.0488** (+18.4%) | 0.0261 | 0.3872 | **0.1048** |

### 4.3 Key Insights from Threshold Calibration
1. **`pos_weight = 94.0` Skews the Decision Boundary:** Because positive change errors are penalized 94× more heavily than background errors during training, model logits are systematically shifted positive. At the uncalibrated default threshold of 0.5, models produce excessive false alarms.
2. **Major Performance Recovery in Sync CNN:** Simply shifting from $\tau = 0.50$ to the validation-selected $\tau^* = 0.830$ increases Sync CNN Test F1 from **0.1102 to 0.1994**, cuts FAR by more than 3× (0.0633 $\to$ 0.0198), and more than doubles Precision (0.0610 $\to$ 0.1313).
3. **Transformer False Alarm Suppression:** Shifting to $\tau^* = 0.730$ cuts Transformer test FAR almost in half (0.1892 $\to$ 0.1048), though precision remains low (0.0261).

---

## 5. Phase D: Controlled Retraining Recommendations & Roadmap

### 5.1 Confirmation of Training Loss Defect
In [`src/training/train_scenario2_dual_stream.py`](file:///C:/Users/Manishym/Researchpapers/satellite%E2%80%91change%E2%80%91detection/projects/multimodal-satellite-change-detection/src/training/train_scenario2_dual_stream.py#L344-L370):
```python
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
loss = criterion(logits, label)  # Unmasked loss over all crop pixels
```
Because `np.nan_to_num(arr, nan=0.0)` replaced missing SAR data with `0.0`, training crops drawn from high-NaN training sites (`L15-0434E` with 78% invalid pixels; `L15-1335E` with 71% invalid pixels) force the models to learn spurious associations between flat zero SAR inputs and building change labels.

---

### 5.2 Controlled Experimental Roadmap (One Factor at a Time)

To improve model stability and scientific defensibility without compromising reproducibility:

```
[Baseline Checkpoints & Metrics] (Preserved & Frozen)
               │
               ▼
   [Experiment S2-1: Masked Loss Training]
   - Mask invalid SAR pixels (weight=0) in BCEWithLogitsLoss
   - Evaluate on both All-Pixel and Valid-Pixel metrics
   - Checkpoint selection using Validation F1 on valid pixels
               │
               ▼
   [Experiment S2-2: Loss Formulation / Weight Recalibration]
   - Investigate pos_weight moderation (e.g. pos_weight=20 or 30 vs 94)
   - Evaluate Compound Loss: BCE + Soft Dice on valid pixels
   - Mitigate early epoch-2 recall-overtriggering collapse
               │
               ▼
   [Experiment S2-3: Input Representation with Validity Channel]
   - Append binary SAR validity mask as an explicit input channel
   - Allows network to distinguish genuine 0.0 backscatter from missing data
               │
               ▼
   [Experiment S2-4: Multi-Seed Verification (Seeds 42, 43, 44)]
   - Measure variance across small 4-site test set
```

#### Priority 1: Implement Masked-Loss Training (Experiment S2-1)
- Create a dedicated experiment directory: `experiments/scenario_2_masked_loss/` and `checkpoints/scenario_2_masked_loss/`.
- Modify crop sampler and training loop to compute pixel-weighted loss:
  $$\mathcal{L} = \frac{\sum_{i} m_i \cdot \ell_i}{\sum_i m_i + \epsilon}, \quad m_i \in \{0, 1\}$$
  where $m_i = 1$ if pixel $i$ has valid SAR at both dates, and $0$ otherwise.
- Keep model architectures identical.

---

## 6. Audit Verdict & Sign-Off

- **Validity Masks Audited:** Yes. Raw NaN patterns established across all 28 sites.
- **Problematic Site Diagnosed:** Yes. False alarms on `L15-1289E` are verified to be algorithmic / probability calibration issues rather than sensor NaNs.
- **Threshold Calibration Completed:** Yes. Validation-only threshold selection demonstrated dramatic performance gains ($\tau^* = 0.83$ yields Sync CNN F1 = 0.1994, FAR = 0.0198).
- **Training Loss Confirmed:** Yes. Existing baseline trained unmasked on NaN-filled zeros.
- **Next Action:** Propose implementing Experiment S2-1 (Masked Loss Training) in a separate experiment directory.
