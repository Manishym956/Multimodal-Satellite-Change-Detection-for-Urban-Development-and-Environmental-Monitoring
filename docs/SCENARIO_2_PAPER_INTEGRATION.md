# Scenario 2 — Paper Integration Notes

Updated against the saved Scenario 2 metrics. The trained split is Strategy B: March 2018 to December 2019, 20 / 4 / 4 sites, one 21-month interval. Labels are a building-probability XOR at 0.5.

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
   7.2 Polarization: no VV-only vs VH-only vs VV+VH result exists (RQ3 is missing)
   7.3 Asynchronous vs synchronous trade-off
   7.4 Limitations (single seed, derived labels, no official split)
8. Conclusion
```

---

## What Scenario 2 contributes

The SEN12-MT archive has 80 sites and monthly files. The trained Scenario 2 split uses 28 of those sites (20 / 4 / 4), not all 80. It uses one 21-month pair, not a multi-gap sweep. Scenario 2 SAR inputs are VV and VH. Change labels are derived building-probability maps, not manual OSCD masks. The OSCD clean-async test F1 of 0.4117 is a single seed and is not evidence that the same gain holds here.

Absolute F1 values are not comparable across the two datasets. An OSCD ranking that places a Transformer next to a CNN also mixes losses: the CNNs used unweighted BCE and the OSCD Transformers used `pos_weight=41.0`. Scenario 2 CNN and Transformer runs both used `pos_weight=94.0`.

---

## Mandatory disclosures in the paper

1. Change labels for SEN12-MT are derived: (buildings_T2 ≥ 0.5) XOR (buildings_T1 ≥ 0.5). This threshold must be stated and its sensitivity acknowledged.
2. The trained pair is T1 = March 2018 and T2 = December 2019, one interval of about 21 months. It is not an earliest-to-latest pairing and it is not a 24-month gap study. RQ2 remains missing.
3. The locked split is 20 train / 4 validation / 4 test sites. It is not a 70/15/15 split.
4. S1 NaN pixels are filled with 0.0 after the validity mask is taken from those NaNs. A zero that was already present stays valid.
5. Training `pos_weight` is 94.0. The class ratio 4,718,348 / 50,387 equals 93.6 when rounded to one decimal place. 93.6 is not the training weight.
6. Scenario 2 uses 4-channel SAR (VV and VH at each date). Asynchronous optical input is 10 Sentinel-2 bands at T1. OSCD SAR is VV only, and OSCD CNNs used unweighted BCE while the OSCD Transformers used `pos_weight=41.0`.
7. The main Scenario 2 baselines are seed 42. Seeds 43 and 44 are a descriptive matched Sync CNN comparison only. They do not estimate a significant seed effect, and they do not complete RQ2 or RQ3.

---

## What Scenario 2 cannot claim

- Cannot claim absolute improvement over Scenario 1 (different datasets).
- Cannot claim that higher F1 on SEN12-MT proves the architecture is better.
- Cannot attribute performance differences solely to data volume without controlling for modality differences.
- Cannot answer the temporal-gap question from the saved runs. Those runs use one 21-month pair. Monthly files exist in the archive, but no multi-gap experiment was trained. RQ2 and RQ3 remain missing, and environmental pseudo-change analysis remains blocked by the building-probability XOR labels.
