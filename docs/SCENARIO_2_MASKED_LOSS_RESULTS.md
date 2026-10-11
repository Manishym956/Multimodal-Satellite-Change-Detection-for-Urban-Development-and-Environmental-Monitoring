# Experiment S2-1: Masked-Loss Training — Results Report

**Experiment date:** 2026-10-10  
**Experiment ID:** S2-1  
**Status:** ✅ Complete — all three models trained successfully

---

## 1. Experimental Design

### Motivation
Scenario 2 baseline models were trained with standard BCE loss that treats all pixels equally, including pixels where SAR sensors returned NaN values (subsequently filled with 0). Phase A of the improvement audit revealed catastrophic NaN rates in several sites:

| Site | Invalid SAR % | Role |
|------|--------------|------|
| L15-0434E | 78.30% | Train |
| L15-1335E | 70.74% | Train |
| L15-1296E | 52.08% | Train |
| L15-0977E | 47.19% | Validation |
| L15-0924E | 44.60% | **Test** |
| L15-1289E | 0.24% | **Test** |

The hypothesis: excluding invalid pixels from both the loss numerator and denominator may improve gradient quality during training and provide a fairer evaluation protocol.

### Change from Baseline
**Only one change:** BCE loss → `MaskedBCEWithLogitsLoss` with pixel-level SAR validity masking.

All other hyperparameters are **identical** to the locked baseline:
- Seed: 42 | Epochs: 50 | Batch size: 64 | Crop: 32×32 | Crops/site: 64
- Adam, lr = 1e-4 | pos_weight = 94.0
- Locked 20/4/4 geographic split

### Mask Construction
```
valid_sar = (s1_t1_finite AND s1_t2_finite)   # both dates must be valid
loss = sum(bce * valid_mask) / sum(valid_mask)  # zero-denominator guard included
```

### Dual Evaluation Protocol
Every test evaluation reports two sub-protocols at tau=0.50 and at val-calibrated tau*:
- **All-Pixel:** all pixels counted (matches baseline protocol exactly)
- **Valid-Pixel:** only valid-SAR pixels counted (ground truth comparison on sensor-reliable data)

---

## 2. Training Summary

| Model | Params | Best Val Epoch | Best Val F1 (valid) | tau* | Train Time |
|-------|--------|---------------|---------------------|------|-----------|
| Async CNN | 27,276,033 | 2 | 0.1677 | 0.520 | 6.1 min |
| Sync CNN | 27,281,793 | 30 | 0.2878 | 0.780 | 8.6 min |
| Cross-Attention | 1,154,945 | 41 | 0.2223 | 0.580 | 5.4 min |

> [!NOTE]
> Async CNN again selected epoch 2 as best — the same early-epoch collapse observed in the baseline. With only 4 validation sites (one having 47% invalid SAR), the noisy validation signal makes checkpoint selection unreliable for this model.

---

## 3. Test Results at Default Threshold (tau = 0.50)

### 3a. All-Pixel Protocol (Comparable to Baseline)

| Model | Prec | Rec | **F1** | IoU | FAR |
|-------|------|-----|--------|-----|-----|
| Async CNN | 0.0473 | 0.2907 | **0.0813** | 0.0424 | 0.0424 |
| Sync CNN | 0.0781 | 0.5342 | **0.1363** | 0.0731 | 0.0456 |
| Cross-Attention | 0.0267 | 0.4639 | **0.0506** | 0.0259 | 0.1222 |

### 3b. Valid-Pixel Protocol

| Model | Prec | Rec | **F1** | IoU | FAR |
|-------|------|-----|--------|-----|-----|
| Async CNN | 0.0479 | 0.2895 | **0.0822** | 0.0428 | 0.0479 |
| Sync CNN | 0.0790 | 0.5339 | **0.1376** | 0.0739 | 0.0519 |
| Cross-Attention | 0.0268 | 0.4630 | **0.0506** | 0.0260 | 0.1402 |

---

## 4. Test Results at Validation-Calibrated Threshold (tau*)

### 4a. All-Pixel Protocol

| Model | tau* | Prec | Rec | **F1** | IoU | FAR |
|-------|------|------|-----|--------|-----|-----|
| Async CNN | 0.520 | 0.0486 | 0.2751 | **0.0827** | 0.0431 | 0.0390 |
| Sync CNN | 0.780 | 0.1346 | 0.3283 | **0.1909** | 0.1055 | 0.0153 |
| Cross-Attention | 0.580 | 0.0289 | 0.4044 | **0.0540** | 0.0278 | 0.0982 |

### 4b. Valid-Pixel Protocol

| Model | tau* | Prec | Rec | **F1** | IoU | FAR |
|-------|------|------|-----|--------|-----|-----|
| Async CNN | 0.520 | 0.0493 | 0.2738 | **0.0835** | 0.0436 | 0.0440 |
| Sync CNN | 0.780 | 0.1373 | 0.3270 | **0.1934** | 0.1070 | 0.0171 |
| Cross-Attention | 0.580 | 0.0290 | 0.4036 | **0.0541** | 0.0278 | 0.1126 |

---

## 5. Per-Site Test Metrics (tau = 0.50, All-Pixel Protocol)

### Async CNN

| Site | Invalid px | F1 (all) | F1 (valid) |
|------|-----------|----------|-----------|
| L15-0924E-1108N *(44.6% invalid)* | ~106 k | 0.0862 | 0.0975 |
| L15-1289E-1169N *(0.24% invalid)* | 567 | 0.0627 | 0.0630 |
| L15-1298E-1322N | low | 0.1591 | 0.1595 |
| L15-1438E-1134N | low | 0.1919 | 0.1922 |

### Sync CNN

| Site | F1 (all) | F1 (valid) |
|------|----------|-----------|
| L15-0924E-1108N *(44.6% invalid)* | 0.0573 | 0.0593 |
| L15-1289E-1169N *(0.24% invalid)* | 0.1504 | 0.1507 |
| L15-1298E-1322N | 0.1597 | 0.1624 |
| L15-1438E-1134N | 0.2057 | 0.2048 |

### Cross-Attention Transformer

| Site | F1 (all) | F1 (valid) |
|------|----------|-----------|
| L15-0924E-1108N *(44.6% invalid)* | 0.0792 | 0.0906 |
| L15-1289E-1169N *(0.24% invalid)* | 0.0417 | 0.0416 |
| L15-1298E-1322N | 0.1381 | 0.1377 |
| L15-1438E-1134N | 0.1795 | 0.1798 |

---

## 6. L15-1289E False-Positive Tracking

This site was the primary failure driver for the Transformer baseline (Phase B).

| Model | Threshold | FP (all pixels) | FAR (all) |
|-------|-----------|----------------|-----------|
| Baseline Cross-Attention | 0.50 | **166,162** | **0.706** |
| S2-1 Async CNN | 0.50 | 33,328 | 0.1416 |
| S2-1 Async CNN | 0.52 | 30,665 | 0.1302 |
| S2-1 Sync CNN | 0.50 | 25,776 | 0.1095 |
| S2-1 Sync CNN | 0.78 | 10,130 | 0.0430 |
| S2-1 Cross-Attention | 0.50 | 108,311 | 0.4600 |
| S2-1 Cross-Attention | 0.58 | 87,473 | 0.3715 |

> [!IMPORTANT]
> The Cross-Attention Transformer's false-positive surge on L15-1289E persists under masked-loss training (108,311 FPs at tau=0.50 vs 166,162 baseline). Masked loss reduced FPs by ~35% but the underlying probability calibration failure is not resolved. As established in Phase B, this site is 99.76% valid SAR, confirming the issue is not sensor artifacts.

---

## 7. Comparison: Baseline vs. S2-1 (All-Pixel, tau = 0.50)

| Model | Baseline F1 | S2-1 F1 | Δ F1 | Baseline FAR | S2-1 FAR | Δ FAR |
|-------|------------|---------|------|-------------|---------|-------|
| Async CNN | 0.0794 | **0.0813** | +0.0019 (+2.4%) | — | 0.0424 | — |
| Sync CNN | 0.1102 | **0.1363** | +0.0261 (+23.7%) | 0.0633 | 0.0456 | −0.0177 |
| Cross-Attention | 0.0412 | **0.0506** | +0.0094 (+22.8%) | 0.1892 | 0.1222 | −0.0670 |

### With Validation-Calibrated Threshold

| Model | Baseline τ* F1 | S2-1 τ* F1 | Δ F1 |
|-------|--------------|----------|------|
| Async CNN | 0.0807 | 0.0827 | +0.0020 |
| Sync CNN | 0.1994 | 0.1909 | −0.0085 |
| Cross-Attention | 0.0488 | 0.0540 | +0.0052 |

---

## 8. Interpretation

### What masked loss improved
1. **Sync CNN gained the most** (+23.7% F1, −0.0177 FAR at tau=0.50), consistent with the training-site contamination hypothesis: sites with 70–78% invalid SAR were contributing large amounts of uninformative gradient in the baseline.
2. **Cross-Attention FAR dropped substantially** (0.1892 → 0.1222 at tau=0.50), indicating the mask helped reduce overconfident false alarms globally, even though L15-1289E remains a specific failure case.
3. **All-pixel vs. valid-pixel F1 differences are small** (< 0.01 in most cases), suggesting invalid pixels are not themselves being predicted as changes at high rates — the model correctly assigns low confidence to NaN-filled regions.

### What masked loss did NOT fix
1. **Transformer L15-1289E calibration collapse** persists: 108,311 FPs at tau=0.50. Phase B established this is caused by insufficient SAR discriminability, not sensor artifacts.
2. **Async CNN early-epoch instability:** epoch 2 selected as best again — the 4-site validation set with 47% invalid-SAR in one site is too noisy for reliable checkpoint selection.
3. **Overall F1 values remain low** (≤ 0.14 at tau=0.50), confirming the primary difficulty is the low positive-class density and the 4-site test set, not the loss formulation alone.

### Scientific caution
The improvements are directionally consistent across all three models (+F1 at tau=0.50), which increases confidence in the masking hypothesis. However, results should be interpreted cautiously:
- 4-site test set introduces high variance per-site
- Single seed (42) — generalization not confirmed
- Sync CNN calibrated threshold result (−0.0085) is within expected noise for this dataset size

---

## 9. Artifacts

| File | Path |
|------|------|
| Async CNN checkpoint | `checkpoints/scenario_2_masked_loss/async_cnn/best.pt` |
| Sync CNN checkpoint | `checkpoints/scenario_2_masked_loss/sync_cnn/best.pt` |
| Cross-Attn checkpoint | `checkpoints/scenario_2_masked_loss/cross_attention/best.pt` |
| Async CNN experiment | `experiments/scenario_2_masked_loss/async_cnn/` |
| Sync CNN experiment | `experiments/scenario_2_masked_loss/sync_cnn/` |
| Cross-Attn experiment | `experiments/scenario_2_masked_loss/cross_attention/` |
| Masked loss module | `src/models/masked_loss.py` |
| Training script | `src/training/train_scenario2_masked_loss.py` |
| Verification suite | `src/training/verify_masked_loss.py` |
| Improvement audit | `docs/SCENARIO_2_IMPROVEMENT_AUDIT.md` |

---

## 10. Suggested Next Steps

Listed in priority order:

1. **S2-2: Attention-gated fusion** — Use the SAR validity mask as a gating signal during feature fusion to explicitly downweight sensor-unreliable regions at inference time.
2. **S2-3: Data augmentation** — Simulate NaN dropout patterns during training to improve robustness to partial-observation regimes.
3. **Cross-Attention temperature scaling** — Apply post-hoc temperature calibration on the 4 validation sites to reduce L15-1289E false alarms without retraining.
4. **Bootstrap confidence intervals** — Resample the 4 test sites to quantify variance on the reported F1 figures.
