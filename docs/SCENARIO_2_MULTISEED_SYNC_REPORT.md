# Scenario 2 Sync CNN Multi-Seed Report (Seeds 42, 43, 44)

**Audit & Completion Date:** October 11, 2026  
**Status:** Completed & Independently Verified  
**Target:** Sync Dual-Stream CNN (`DualStreamUNet`, 27,281,793 parameters)  
**Comparison:** Standard weighted BCE (`pos_weight=94.0`, all pixels) vs. masked weighted BCE (`pos_weight=94.0`, valid SAR pixels only). Seeds 43 and 44 are the matched pair: identical initialization within each seed, identical site-specific crop schedule, identical DataLoader sampling order, and the same checkpoint rule. Seed 42 used a different checkpoint-selection rule and is reported separately.

---

## 1. Executive Summary

In Experiment S2-1, the preliminary Seed 42 run recorded Sync CNN all-pixel test F1 of **0.1102** for standard weighted BCE and **0.1363** for masked weighted BCE (+0.0261 on the masked-minus-standard definition). The standard-loss checkpoint peaked at Epoch 21 and the masked-loss checkpoint peaked at Epoch 30, so that difference is confounded with checkpoint selection.

Seeds 43 and 44 are a fully matched Sync CNN comparison, with:
1. Identical model weight initialization within each seed pair.
2. Identical pre-generated deterministic crop schedules based on actual site dimensions.
3. Predeclared checkpoint-selection rule: Highest Valid-Pixel Validation F1 (`val_f1_valid`).
4. Dual-protocol evaluation (All-Pixel and Valid-Pixel) at default threshold $\tau = 0.50$ and validation-calibrated $\tau^*$.

### Key Findings
1. **Matched checkpoint epochs (seeds 43 and 44 only):**
   - In **Seed 43**, both the standard weighted BCE and masked weighted BCE models peaked at **Epoch 30** (`val_f1_valid` = 0.2932 vs. 0.2877).
   - In **Seed 44**, both models peaked at **Epoch 43** (`val_f1_valid` = 0.2746 vs. 0.2744).
   - In these two runs, the selected epochs coincide. That describes these runs only.
2. **Descriptive all-pixel F1 from the two matched seeds** ($\tau = 0.50$):
   - Paired difference = masked-loss all-pixel F1 minus standard weighted BCE all-pixel F1. This is the order used in `docs/paper/02_results_v2.md` (there the mean is rounded to $-0.0139$).
   - Seed 43: $0.1236 - 0.1327 = -0.0091$.
   - Seed 44: $0.1141 - 0.1327 = -0.0186$.
   - Mean of those two paired differences: $-0.01385$.
   - These are descriptive results from two matched seeds. They do not establish statistical significance, equivalence, or a general advantage for either loss.
3. **Seed 42, not pooled with the matched pair:**
   - Standard weighted BCE all-pixel F1 = **0.1102** (epoch 21, all-pixel validation selection).
   - Masked weighted BCE all-pixel F1 = **0.1363** (epoch 30, valid-pixel validation selection).
   - Paired difference on that same definition (masked minus standard) = $+0.0261$.
   - Seed 42 used a different checkpoint-selection rule for the two losses, and the selected epochs differ (30 vs. 21). That difference stays outside the matched comparison.
4. Masked loss keeps invalid SAR pixels out of the training objective. The matched-seed all-pixel F1 differences are negative, so this report does not conclude that masked loss leaves overall test F1 unchanged.

---

## 2. Multi-Seed Quantitative Comparison Table

*All models evaluated on the locked SpaceNet 7 test sites (Strategy B). Both losses use `pos_weight=94.0`. Seed 42 is listed for the recorded values only; it is not part of the matched comparison.*

| Seed & Model Variant | Peak Ep | Val F1 (valid) | Test Prec (all) | Test Rec (all) | Test F1 (all @ 0.50) | Test FAR (all) | Test F1 (valid) | Calibrated $\tau^*$ | Test F1 (all @ $\tau^*$) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Seed 42 Baseline** | 21 | 0.2663 | 0.0610 | 0.5680 | **0.1102** | 0.0633 | 0.1102 | 0.50 | 0.1102 |
| **Seed 42 Masked** | 30 | 0.2878 | 0.0781 | 0.5342 | **0.1363** | 0.0456 | 0.1376 | 0.78 | 0.1909 |
| **Seed 43 Baseline** | 30 | 0.2932 | 0.0757 | 0.5349 | **0.1327** | 0.0472 | 0.1342 | 0.79 | 0.1943 |
| **Seed 43 Masked** | 30 | 0.2877 | 0.0693 | 0.5719 | **0.1236** | 0.0556 | 0.1257 | 0.83 | 0.1912 |
| **Seed 44 Baseline** | 43 | 0.2746 | 0.0751 | 0.5721 | **0.1327** | 0.0510 | 0.1337 | 0.86 | 0.2180 |
| **Seed 44 Masked** | 43 | 0.2744 | 0.0636 | 0.5572 | **0.1141** | 0.0594 | 0.1180 | 0.91 | 0.2113 |

Seed 42 baseline validation F1 0.2663 is the all-pixel validation F1 recorded for that checkpoint. The "Val F1 (valid)" header applies to the matched seeds and to the Seed 42 masked run.

---

## 3. Descriptive paired differences

Seeds 43 and 44 are the matched comparison. The difference is masked-loss all-pixel F1 minus standard weighted BCE all-pixel F1 at $\tau = 0.50$. The mean is the arithmetic mean of those two differences. This is a description of two seeds, not a significance test and not evidence that the losses are equivalent.

| Metric ($\tau = 0.50$, All-Pixel) | Standard weighted BCE | Masked weighted BCE | Paired difference (masked $-$ standard) |
|---|:---:|:---:|:---:|
| **Seed 43 F1** | 0.1327 | 0.1236 | $-0.0091$ |
| **Seed 44 F1** | 0.1327 | 0.1141 | $-0.0186$ |
| **Mean of the two paired differences** |  |  | **$-0.01385$** |

Seed 42 remains a separate, unmatched record. Its standard-loss checkpoint was selected on all-pixel validation F1, and its masked-loss checkpoint was selected on valid-pixel validation F1.

| Metric ($\tau = 0.50$, All-Pixel) | Standard weighted BCE | Masked weighted BCE | Difference (masked $-$ standard) |
|---|:---:|:---:|:---:|
| **Seed 42 F1 (not matched)** | 0.1102 | 0.1363 | $+0.0261$ |
