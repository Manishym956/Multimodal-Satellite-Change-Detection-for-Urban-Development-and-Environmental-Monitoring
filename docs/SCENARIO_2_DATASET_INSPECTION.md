# Scenario 2 — Dataset Inspection Report

Dataset: SEN12-MT Urban Mapping (based on SpaceNet 7 sites)
Dataset root: `datasets/sen12_mt_urbanmapping/multimodal_cd_dataset/`
Inspection date: 2026-10-06
Status: **SUPPORTED WITH MODIFICATIONS** — see Section 13 and Final Recommendation.

---

## 1. Directory structure  ✅ VERIFIED

```
datasets/sen12_mt_urbanmapping/
└── multimodal_cd_dataset/
    ├── metadata.json          ← per-site temporal availability table
    ├── bad_data.json          ← known-bad timestep indices per site
    └── {AOI_ID}/              ← one directory per geographic site (80 sites)
        ├── s1/
        │   └── s1_{AOI_ID}_{YYYY}_{MM}.tif
        ├── s2/
        │   └── s2_{AOI_ID}_{YYYY}_{MM}.tif
        └── buildings/
            └── buildings_{AOI_ID}_{YYYY}_{MM}.tif
```

AOI ID format: `L15-{EASTING}E-{NORTHING}N_{TILE_X}_{TILE_Y}_13`
These are SpaceNet 7 tile identifiers (WebMercator zoom level 15).

---

## 2. Number of sites / cities / scenes  ✅ VERIFIED

| Statistic | Value |
|---|---|
| Total geographic sites | 80 |
| Timestamps per site (min / max / mean) | 18 / 26 / 23.6 |
| Year range | 2017, 2018, 2019, 2020 |
| Total timesteps across all sites | 1,889 |
| Timesteps with S1 available | 1,843 / 1,889 |
| Timesteps with S2 available | 1,494 / 1,889 |
| Known-bad S1 timestep indices | 46 |
| Known-bad S2 timestep indices | 395 |

80 sites is substantially more than OSCD (24 cities). ✅

---

## 3. Number of temporal observations  ✅ VERIFIED

The dataset is **multi-temporal** (not bitemporal). Each site has approximately monthly observations from early 2018 to January 2020 (some sites include late 2017). There is no fixed "T1 / T2" change-detection pair in the original data. Bitemporal pairs must be selected from the available timesteps.

Dates are encoded in filenames as `{YYYY}_{MM}` (year and month). Acquisition dates are known with month-level precision.

---

## 4. Sentinel-1 channels  ✅ VERIFIED

| Property | Value |
|---|---|
| Bands | 2 (VV + VH) |
| Band descriptions | ('VV', 'VH') from rasterio |
| Data type | float32 |
| Value range (valid pixels) | [0.0, 1.0] — already normalised |
| Spatial resolution | 10 m |
| CRS | WGS 84 / Pseudo-Mercator (EPSG:3857, stored as LOCAL_CS) |
| Tile size | ~489 × 488 pixels (varies slightly) |
| NaN pixels | Present — bad pixels indicated in bad_data.json |
| No-data value | None stored; NaN is used |

**VH is present.** This enables 4-channel SAR input (VV T1, VH T1, VV T2, VH T2) in Scenario 2.

---

## 5. Sentinel-2 bands  ✅ VERIFIED

| Property | Value |
|---|---|
| Bands | 10 |
| Band descriptions | B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 |
| Data type | float32 |
| Value range | [0.0, 1.0] — **already normalised; do NOT divide by 10000** |
| Spatial resolution | 10 m (all bands resampled to 10 m) |
| CRS | WGS 84 / Pseudo-Mercator |
| Tile size | 489 × 488 (matches S1 grid exactly) |
| NaN pixels | None observed in sampled files |

Note: OSCD had 13 S2 bands (including B01, B09, B10). This dataset has 10 bands (missing B01, B09, B10). The Scenario 2 optical input is **10 channels, not 13**.

---

## 6. Label format  ✅ VERIFIED — CRITICAL FINDING

The `buildings/` folder contains **per-timestep building probability maps**, not binary change masks.

| Property | Value |
|---|---|
| Format | Single-band float32 GeoTIFF |
| Value range | [0.0, 1.0] (continuous building probability) |
| Alignment | Same grid as S1 and S2 (489 × 488, 10 m) |
| Coverage | Present for all 80 sites, all timesteps |

**There are NO pre-computed binary change masks.** Change labels must be derived by comparing building maps across two timesteps:

```python
change_mask = (buildings_T2 >= 0.5).astype(int) XOR (buildings_T1 >= 0.5).astype(int)
```

This derivation is scientifically defensible: if a pixel has building probability ≥ 0.5 at T2 but < 0.5 at T1, it represents new urban construction.

Observed change fractions (T1=earliest unmasked → T2=latest unmasked):
- Range across 8 sampled sites: 0.05% – 1.43%
- Typical: ~0.3–1.0%

This is even sparser than OSCD (~2.4% change pixels), so `pos_weight` will be substantially higher than 41. Must be computed from training split.

---

## 7. Spatial dimensions  ✅ VERIFIED

| Property | Value |
|---|---|
| Tile size | ~489 × 488 pixels (slight variation across sites) |
| All modalities co-registered | Yes (same grid for S1, S2, buildings) |
| Sufficient for 32×32 crops | Yes — 489 × 488 >> 32 |

---

## 8. CRS / georeferencing  ✅ VERIFIED

All files use WGS 84 / Pseudo-Mercator at 10 m. S1 and S2 are co-registered to the same pixel grid per site. No reprojection needed.

---

## 9. Temporal metadata  ✅ VERIFIED

Acquisition months are encoded in filenames (`YYYY_MM`). This is month-level precision. Individual SAR overpass dates within a month are not provided. The `metadata.json` records `year` and `month` per timestep, and whether S1, S2, and buildings data are available.

For the asynchronous experiment: the historical-vs-target temporal ordering can be established using year/month metadata. If T1 is the earliest unmasked timestep and T2 is the latest, T1 is verifiably earlier than T2.

---

## 10. Missing / corrupt files  ✅ VERIFIED

`bad_data.json` explicitly identifies bad timestep indices for S1 and S2 per site. These must be excluded at data loading time. Key counts:
- 46 bad S1 timestep indices across all sites
- 395 bad S2 timestep indices across all sites (S2 has more cloud/mask issues)

NaN pixels are present in S1 files even for non-bad timesteps (sparse boundary NaN). Must be handled in the loader (fill with 0.0 or mask out before loss computation).

No zero-byte or missing files detected in directory inspection.

---

## 11. Class balance  ⚠️ REQUIRES COMPUTATION FROM TRAINING SPLIT

From 8 sampled sites (earliest → latest), change pixel fraction: 0.05% – 1.43%.

This is considerably sparser than OSCD (~2.4%). Estimated pos_weight will be substantially higher than the OSCD value of 41.

**pos_weight must be recomputed from the actual training split after the split is defined.** Do not reuse pos_weight=41 from OSCD.

Rough estimate based on ~0.5% change fraction: pos_weight ≈ (99.5 / 0.5) = ~200. This will be verified at split time.

---

## 12. Train / validation / test information  ⚠️ NO OFFICIAL SPLIT

No `train.txt`, `test.txt`, or official split file is present in the dataset. A site-level split must be defined manually.

Proposed split (seed 42, site-level, no overlap):
- 56 train sites (70%)
- 12 validation sites (15%)
- 12 test sites (15%)

The test split must be locked before any training or hyperparameter tuning. The exact site assignment will be documented in `experiments/scenario_2/configs/split_definition.json` before any training run.

---

## 13. Historical vs target optical — scientific validity  ✅ SUPPORTED WITH CONSTRAINTS

The multi-temporal structure enables a principled asynchronous setup:

- T1 = earliest available unmasked timestep per site (e.g., 2018-01)
- T2 = latest available unmasked timestep per site (e.g., 2020-01)
- Historical optical input = S2 at T1 (verifiably before the change period)
- SAR input = S1 at T1 and S1 at T2 (both required for change detection)
- Contemporary optical at T2 = withheld from asynchronous models
- Ground-truth change mask = derived from (buildings_T2 >= 0.5) XOR (buildings_T1 >= 0.5)

This is scientifically valid. T1 is verifiably earlier than T2 by file metadata. The typical temporal gap is approximately 24 months (2018-01 to 2020-01).

**Constraint:** The change label is derived from building probability maps, not from a separately validated change annotation. The derivation procedure must be documented and the threshold (0.5) must be stated explicitly in the paper.

---

## 14. VV / VH availability  ✅ VERIFIED

Both VV and VH are present in the S1 files (confirmed by rasterio band descriptions). Scenario 2 will use 4-channel SAR input: S1-VV-T1, S1-VH-T1, S1-VV-T2, S1-VH-T2.

This is a difference from Scenario 1 (VV only). The model `sar_encoder` input must be adjusted to 4 channels.

---

## 15. Storage requirements  ✅ VERIFIED

| Item | Size |
|---|---|
| Total GeoTIFFs on disk | 18.45 GB |
| Archive files remaining | None detected |
| Available disk space | ~65 GB |

Sufficient disk space is available. Large imagery is already excluded by `.gitignore` (`*.tif`).

---

## Final Recommendation

**Status: SUPPORTED WITH MODIFICATIONS**

The SEN12-MT Urban Mapping dataset supports a scientifically valid Scenario 2 experiment with the following documented modifications to the original plan:

| Original assumption | Actual situation | Required change |
|---|---|---|
| Binary change masks provided | Building probability maps, must derive change | Derive: (bld_T2 ≥ 0.5) XOR (bld_T1 ≥ 0.5) |
| SAR: VV only | VV + VH both available | Use 4-channel SAR (2 bands × 2 dates) |
| S2: 13 bands | 10 bands (B2–B8A, B11, B12; no B1, B9, B10) | Use 10-channel optical input |
| S2 normalisation: divide by 10000 | Already normalised to [0,1] | Do NOT divide by 10000 |
| pos_weight ≈ 41 | Change fraction ~0.3–1.5%, pos_weight ≈ 100–300 | Recompute from training split |
| Official train/val/test split | No official split | Define site-level split (seed 42, 70/15/15%) |
| NaN-free data | NaN present in S1 | Fill NaN with 0.0 in loader |

These modifications are all implementable without compromising scientific validity. The experiment is defensible.
