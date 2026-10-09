# Scenario 2 — Hyperparameter Sweep Design

Status: READY TO PLAN — dataset inspected, modalities confirmed.
No sweep will run until dataset loader and split are implemented and verified.

---

## Confirmed frozen hyperparameters

These will NOT be swept. Consistent with Scenario 1 unless noted.

| Parameter | Value | Note |
|---|---|---|
| Seed | 42 | Reproducibility |
| Batch size | 64 | GPU memory (RTX 4060 8GB) |
| Crop size | 32 × 32 | Same as Scenario 1 |
| Crops per site per epoch | 64 | Same as Scenario 1 |
| Optimizer | Adam | Same as Scenario 1 |
| Threshold | 0.5 | Same as Scenario 1 |
| Epochs | 50 | Starting point; extend to 100 if not converged |
| Checkpoint selection | Best validation F1 | Same as Scenario 1 |
| last.pt | Unconditional save every epoch | Same as Scenario 1 |
| SAR channels | 4 (VV+VH × T1+T2) | VH now available |
| Optical channels | 10 (B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12 at T1) | 10-band S2 |
| S2 normalisation | [0,1] already; no divide by 10000 | Dataset-specific |
| S1 NaN handling | Fill with 0.0 | Dataset-specific |

---

## pos_weight

pos_weight is NOT freely tunable. It is computed from the training split class balance:

```
pos_weight = no_change_pixels / change_pixels  (training split only)
```

Based on inspection of 8 sites, estimated ~100–300. The sweep will also test ×0.5 and ×2.0 as sensitivity checks, but the computed value is the primary setting.

---

## Candidate sweep values

| Hyperparameter | Candidates | Notes |
|---|---|---|
| Learning rate | 5e-5, **1e-4**, 2e-4 | Bold = Scenario 1 default |
| Transformer self-attn blocks | **2**, 3 | Per modality |
| Transformer cross-attn blocks | **1** | Keep fixed |
| Attention heads | **4**, 8 | |
| pos_weight sensitivity | ×0.5, **×1.0**, ×2.0 | Applied to computed value |

---

## Proposed sweep (10 experiments)

One-factor-at-a-time around the Scenario-1 reference configuration.

| Experiment ID | Model | LR | SA blocks | Heads | pos_weight | Note |
|---|---|---|---|---|---|---|
| S2-ref-sync | Sync Dual-Stream CNN | 1e-4 | N/A | N/A | computed | Sync baseline reference |
| S2-ref-async-cnn | Clean Async CNN | 1e-4 | N/A | N/A | computed | Async CNN reference |
| S2-ref-xattn | Cross-Attn Transformer | 1e-4 | 2 | 4 | computed | Transformer reference |
| S2-lr-low | Cross-Attn Transformer | 5e-5 | 2 | 4 | computed | Lower LR |
| S2-lr-high | Cross-Attn Transformer | 2e-4 | 2 | 4 | computed | Higher LR |
| S2-blocks-3 | Cross-Attn Transformer | 1e-4 | 3 | 4 | computed | More SA blocks |
| S2-heads-8 | Cross-Attn Transformer | 1e-4 | 2 | 8 | computed | More heads |
| S2-pw-low | Cross-Attn Transformer | 1e-4 | 2 | 4 | computed × 0.5 | Lower pos_weight sensitivity |
| S2-pw-high | Cross-Attn Transformer | 1e-4 | 2 | 4 | computed × 2.0 | Higher pos_weight sensitivity |
| S2-cnn-lr-low | Clean Async CNN | 5e-5 | N/A | N/A | computed | CNN LR sensitivity |

Total: 10 experiments. All evaluated on validation set only.

---

## GPU and runtime estimate

| Model | Approx train time (50 epochs) | Approx VRAM |
|---|---|---|
| Clean Async CNN (14-ch) | ~5–8 min | ~2–3 GB |
| Cross-Attn Transformer (14-ch) | ~5–8 min | ~1–2 GB |
| Sync Dual-Stream CNN (24-ch) | ~5–8 min | ~2–3 GB |

Estimated total sweep: ~10 × 7 min = ~70 minutes on RTX 4060. Feasible in a single session.

Note: These are rough estimates based on Scenario 1 timing (~5–7 min per 50-epoch run at 640 crops/epoch). Actual time depends on site count, effective crops, and data loading speed.

---

## Selection procedure

1. Compute pos_weight from training split. Record in `experiments/scenario_2/configs/`.
2. Define and lock the test split. Do not examine test labels.
3. Run the 10 sweep experiments.
4. Select the configuration with the highest validation F1.
5. Lock the configuration. No further tuning.
6. Run final test evaluation once with the locked configuration.
7. Report all test metrics.

---

## Constraint

The CNN learning rate sweep (S2-cnn-lr-low) is included as a single sensitivity check. The transformer sweep (S2-lr-low, S2-lr-high, S2-blocks-3, S2-heads-8, S2-pw-low, S2-pw-high) is more extensive because the transformer was more sensitive to training regime in Scenario 1.

If S2-ref-xattn already exceeds S2-ref-async-cnn on validation F1, no further transformer tuning is required. Confirm and lock immediately.
