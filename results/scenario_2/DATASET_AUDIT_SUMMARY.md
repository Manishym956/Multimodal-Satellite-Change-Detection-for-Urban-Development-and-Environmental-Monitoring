# Scenario 2 — Dataset Audit Summary

Date: 2026-10-06
Dataset: SEN12-MT Urban Mapping
Status: **SUPPORTED WITH MODIFICATIONS**

---

## Dataset location

`datasets/sen12_mt_urbanmapping/multimodal_cd_dataset/`

---

## Verified structure

```
80 geographic sites (SpaceNet 7 tile IDs)
├── s1/     — Sentinel-1 GeoTIFFs, monthly, 2017–2020
├── s2/     — Sentinel-2 GeoTIFFs, monthly, 2017–2020
└── buildings/ — Building probability maps, monthly, 2017–2020
metadata.json   — timestep availability per site
bad_data.json   — known-bad timestep indices
```

---

## Verified modalities

| Modality | Bands | Dtype | Range | Resolution |
|---|---|---|---|---|
| Sentinel-1 | 2 (VV + VH) | float32 | [0, 1] | 10 m |
| Sentinel-2 | 10 (B2–B8A, B11, B12) | float32 | [0, 1] | 10 m |
| Buildings label | 1 (probability) | float32 | [0, 1] | 10 m |

All three modalities are co-registered to the same 489×488 pixel grid per site.

---

## Change label situation

**Pre-computed binary change masks are NOT provided.**

The `buildings/` folder contains per-timestep building probability maps. Binary change masks must be derived:

```
change_mask = (buildings_T2 >= 0.5) XOR (buildings_T1 >= 0.5)
```

where T1 = earliest unmasked timestep, T2 = latest unmasked timestep per site.

Observed change fraction: ~0.3–1.5% per site (sparser than OSCD ~2.4%).
Required pos_weight: ~100–300 (must be computed from training split).

---

## Asynchronous experiment validity

**SUPPORTED.** The multi-temporal structure allows:
- T1 optical (historical): S2 at earliest unmasked timestep
- SAR T1+T2: S1 at T1 and T2
- Contemporary optical T2: withheld from async models

Temporal ordering verified: T1 year/month < T2 year/month from metadata.

---

## Key differences from Scenario 1 (OSCD)

| Item | OSCD | SEN12-MT |
|---|---|---|
| SAR channels | 2 (VV × 2 dates) | 4 (VV+VH × 2 dates) |
| Optical channels | 13 bands | 10 bands |
| S2 preprocessing | Divide by 10000 | Already [0,1]; no division needed |
| Change labels | Binary masks provided | Derived from building probability |
| Official split | Yes | No — site-level split required |
| Sites/cities | 24 | 80 |

---

## Next steps (implementation order)

1. Define and lock train/val/test split → `experiments/scenario_2/configs/split_definition.json`
2. Compute pos_weight from training split → `experiments/scenario_2/configs/class_balance.json`
3. Implement `src/datasets/scenario2_dataset.py`
4. Implement `src/training/train_scenario2_dual_stream.py`
5. Implement `src/training/train_scenario2_cross_attention.py`
6. Run verification checks (--verify-only) on all three training scripts
7. Run hyperparameter sweep (validation only)
8. Lock configuration
9. Run final test evaluation once

---

## Blockers requiring decision

None at this stage. All inspection items are resolved. The experiment is implementable.

One design decision to confirm before implementation:
**Change label T1/T2 selection:** Using earliest + latest unmasked non-masked timestep per site is the simplest and most defensible approach. An alternative is to fix the pair at 2018-01 → 2020-01 globally (but some sites may be masked at those months). The simplest approach (earliest/latest unmasked) is recommended and should be confirmed before implementing the loader.
