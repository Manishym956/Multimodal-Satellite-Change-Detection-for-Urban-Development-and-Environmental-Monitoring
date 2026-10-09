# Scenario 2 — Paper Integration Notes

Updated after dataset inspection. No results exist yet.

---

## Dataset confirmed for Scenario 2

SEN12-MT Urban Mapping: 80 sites, monthly S1+S2+buildings, 2017–2020.
Key differences from OSCD (Scenario 1):

| Property | OSCD (S1) | SEN12-MT (S2) |
|---|---|---|
| Sites | 24 cities | 80 sites |
| SAR | VV only (2ch) | VV+VH (4ch) |
| Optical | 13 bands | 10 bands |
| Temporal | Bitemporal | Multi-temporal (monthly) |
| Change labels | Binary masks (provided) | Derived from building prob. maps |
| S2 normalisation | Divide by 10000 | Already [0,1] |

---

## Paper structure

```
1. Introduction
2. Related work
3. Data
   3.1 Scenario 1: OSCD benchmark (24 cities, bitemporal, binary change masks)
   3.2 Scenario 2: SEN12-MT Urban Mapping (80 sites, multi-temporal, derived change labels)
4. Methods
   4.1 Asynchronous input design
   4.2 Change label derivation (SEN12-MT)
   4.3 Model architectures
       Synchronous Dual-Stream CNN
       Clean Asynchronous Dual-Stream CNN
       SAR-guided Cross-Attention Transformer
   4.4 Training protocol
   4.5 Evaluation metrics
5. Scenario 1 results (OSCD)
6. Scenario 2 results (SEN12-MT)
7. Discussion
   7.1 Architecture ranking across datasets
   7.2 Effect of VH polarization
   7.3 Asynchronous vs synchronous trade-off
   7.4 Limitations (single seed, derived labels, no official split)
8. Conclusion
```

---

## What Scenario 2 contributes

Scenario 2 tests whether Scenario 1 findings generalise to a:
- Larger dataset (80 vs 24 sites)
- More temporally diverse dataset (monthly over ~2 years vs 2 time points)
- Dataset with VH SAR available
- Dataset with derived rather than manually annotated change labels

The valid cross-scenario comparison is the relative ranking of architectures within each scenario, not absolute F1 values.

---

## Mandatory disclosures in the paper

1. Change labels for SEN12-MT are derived: (buildings_T2 ≥ 0.5) XOR (buildings_T1 ≥ 0.5). This threshold must be stated and its sensitivity acknowledged.
2. T1 and T2 are defined as earliest and latest unmasked timesteps per site. The typical temporal gap is approximately 24 months.
3. No official train/val/test split exists. A site-level split (70/15/15%, seed 42) was defined before training.
4. S1 NaN pixels are filled with 0.0.
5. pos_weight for BCEWithLogitsLoss was computed from the training split change fraction (expected ~100–300, exact value TBD).
6. Scenario 2 uses 4-channel SAR (VV+VH × 2 dates) and 10-channel S2, different from Scenario 1.
7. Results are single-seed (seed 42). Variance across seeds is not estimated.

---

## What Scenario 2 cannot claim

- Cannot claim absolute improvement over Scenario 1 (different datasets).
- Cannot claim that higher F1 on SEN12-MT proves the architecture is better.
- Cannot attribute performance differences solely to data volume without controlling for modality differences.
- Cannot answer the temporal-gap hypothesis from the original research question (month-level dates available, but SAR scene-level dates within a month are not).
