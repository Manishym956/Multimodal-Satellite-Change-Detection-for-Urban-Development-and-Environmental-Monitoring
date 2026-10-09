# Scenario 2 — Temporal Pairing Feasibility Analysis

Dataset: SEN12-MT Urban Mapping (80 sites, monthly 2017–2020)
Analysis date: 2026-10-06
Source script: `src/training/temporal_pairing_analysis.py`

---

## Summary recommendation

**Use Strategy B: Fixed pair 2018-03 → 2019-12 (21-month interval).**

This is the best-coverage fixed-pair strategy among all candidates evaluated, with 32 sites fully eligible for the asynchronous configuration and 28 for the synchronous, while maintaining a consistent, documentable temporal interval across all sites. It is preferred over Strategy C (earliest/latest) because it gives uniform temporal intervals — a scientifically cleaner comparison.

See Section 6 for full rationale.

---

## 1. Strategies evaluated

Three selection strategies were evaluated using `metadata.json` and `bad_data.json` exclusively (no raster loading for coverage counts). Modality availability is determined by:
- `metadata.s1 = True` AND timestep index not in `bad_data[site].S1` AND `masked = False`
- `metadata.s2 = True` AND timestep index not in `bad_data[site].S2` AND `masked = False`
- `metadata.buildings = True` AND `masked = False`

**Strategy A — Fixed 2018-01 → 2020-01 (24 months)**
**Strategy B — Evaluated five fixed near-24-month candidates**
**Strategy C — Earliest valid / Latest valid per site (variable interval)**

---

## 2. Coverage counts by strategy

### Strategy A: Fixed 2018-01 → 2020-01

| Criterion | Sites (out of 80) |
|---|---|
| Buildings valid at T1 AND T2 | 36 |
| S1 usable at T1 AND T2 | 47 |
| S2 usable at T1 | 47 |
| Full ASYNC eligible (bld+S1+S2 at T1, bld+S1 at T2) | **27** |
| Full SYNC eligible (adds S2 at T2) | **25** |
| Excluded | 53 (async) |

Only 27 sites are usable under Strategy A. The primary exclusion cause is missing buildings labels at T2 (2020-01 is the last available month and not all sites have buildings data at that exact timestamp).

### Strategy B: Fixed pair candidates

| Pair | Gap | Buildings both | S1 both | S2 at T1 | Async eligible | Sync eligible |
|---|---|---|---|---|---|---|
| 2018-01 → 2020-01 | 24 mo | 36 | 47 | 47 | 27 | 25 |
| 2018-02 → 2020-01 | 23 mo | 42 | 55 | 52 | 31 | 28 |
| **2018-03 → 2019-12** | **21 mo** | **41** | **55** | **60** | **32** | **28** |
| 2018-01 → 2019-12 | 23 mo | 39 | 50 | 47 | 31 | 26 |
| 2018-02 → 2019-12 | 22 mo | 43 | 56 | 52 | 31 | 28 |

**2018-03 → 2019-12 gives the best async coverage (32 sites).** It also ties or beats alternatives on sync coverage (28 sites). The 21-month interval is consistent and documentable.

The reason 2018-03 at T1 outperforms 2018-01/02 is that S2 availability at T1 is better in early spring than in January/February (likely reduced cloud coverage or acquisition gaps in the original data).

### Strategy C: Earliest/latest valid per site

| Criterion | Value |
|---|---|
| Full ASYNC eligible | 59 / 80 |
| Full SYNC eligible | 51 / 80 |
| Temporal gap (months) | min=12, max=26, mean=23.1, median=24 |

Strategy C maximises site coverage (59 async, 51 sync) but introduces variable temporal intervals (12–26 months). The variable gap creates a confound: sites with longer gaps may show more change simply because more time has passed, not because the model is better. This is a scientific risk for a paper that compares architectures.

---

## 3. Per-site eligibility (Strategy A vs C comparison)

See `docs/SCENARIO_2_SITE_ELIGIBILITY.csv` for the full 80-site table.

Sites excluded from Strategy A that would be included in Strategy C:
- Many sites have buildings/S1/S2 available in March 2018 but not January 2018.
- Strategy B (2018-03 → 2019-12) recovers most of these.

Sites excluded from all strategies (Strategy C also fails):
- Sites with pervasive bad data (e.g. `L15-1749E-1266N_6997_3126_13`: S1 bad at 14 of 21 timesteps).
- Sites with extreme S2 unavailability (e.g. `L15-0571E-1075N_2287_3888_13`: S2 bad at 17 of 20 timesteps).

---

## 4. Label validity: building probability change masks

**Raster-level checks performed on all 27 Strategy A async-eligible sites.**

### 4.1 Change fraction at threshold 0.5

| Statistic | Value |
|---|---|
| Minimum | 0.051% |
| Maximum | 4.513% |
| Mean | 1.111% |
| Median | 0.677% |
| Sites with zero positive pixels | **0** |

All 27 checked sites have at least some positive change pixels at threshold 0.5. The change class is sparse (median 0.677%) but never zero.

### 4.2 Threshold sensitivity (total positive pixels across 27 sites)

| Threshold | Total positive pixels |
|---|---|
| 0.3 | 91,229 |
| 0.4 | 81,106 | 
| **0.5** | **71,564** |
| 0.6 | 62,632 |
| 0.7 | 54,122 |

The change pixel count decreases roughly 10–12% per 0.1 threshold increment. The threshold 0.5 is the natural midpoint of the [0,1] probability range and is consistent with general practice. It is the recommended threshold.

### 4.3 pos_weight estimate

From 27 Strategy A sites:
- Total no-change pixels: 6,371,469
- Total change pixels: 71,595
- **Estimated pos_weight ≈ 89.0**

This is approximately twice the OSCD Scenario 1 value (41.0), reflecting the sparser change class. The exact pos_weight must be recomputed from the actual training split (56 sites) before training.

### 4.4 Label validity caveat (mandatory disclosure)

The `buildings/` folder contains **per-timestep building probability maps**, not independently validated binary change masks. The change label is derived as:

```
change_mask = (buildings_T2 >= 0.5).astype(int) XOR (buildings_T1 >= 0.5).astype(int)
```

This proxy captures building footprint changes between two monthly composites. It does not capture:
- Demolition followed by reconstruction within the interval (would show no change)
- Temporary construction artefacts
- Changes in non-building urban infrastructure (roads, parking)

This limitation must be stated in the paper. The label is a reasonable proxy for urban construction change but is not a directly annotated ground truth.

---

## 5. Temporal consistency for asynchronous vs synchronous models

Under the recommended Strategy B (2018-03 → 2019-12):

| Model type | T1 input | T2 input | Change label source |
|---|---|---|---|
| Async (Clean CNN, Cross-Attn) | S1 VV+VH + S2 at 2018-03 | S1 VV+VH only at 2019-12 | (bld_2019-12 ≥ 0.5) XOR (bld_2018-03 ≥ 0.5) |
| Synchronous CNN baseline | S1 VV+VH + S2 at 2018-03 | S1 VV+VH + S2 at 2019-12 | Same |

The **S2 image at T2 (2019-12) is never loaded or passed to the asynchronous models.** The synchronous baseline receives it as its "contemporary" optical input. This is the critical distinction between async and sync.

The change label uses the same T1/T2 endpoints for both model types, ensuring the task is identical.

---

## 6. Recommendation and rationale

**Recommended strategy: Strategy B — Fixed 2018-03 → 2020-01**

Wait — 2018-03 → 2019-12 vs 2018-02 → 2020-01 tie at 28 sync sites but 2018-03→2019-12 has 32 async sites. Let's verify the exact recommendation:

| Pair | Async | Sync | Gap |
|---|---|---|---|
| 2018-03 → 2019-12 | **32** | 28 | 21 mo |
| 2018-02 → 2019-12 | 31 | 28 | 22 mo |
| 2018-02 → 2020-01 | 31 | 28 | 23 mo |

**Final recommendation: 2018-03 (March) → 2019-12 (December), a 21-month interval.**

Rationale:
1. **Highest async coverage**: 32 sites, vs 27–31 for other fixed pairs.
2. **Consistent temporal interval**: Every site uses the same two months. No variable-gap confound.
3. **21 months is scientifically meaningful**: Sufficient for urban construction change to be detectable.
4. **Documentable**: The pair is stated once, clearly, in the paper.
5. **Strategy C rejected**: Despite higher coverage (59 sites), variable 12–26 month gaps introduce a temporal confound that complicates architecture comparison.
6. **Strategy A rejected**: Only 27 async sites — too small for robust evaluation. Buildings data is sparse in January 2020.

### Fallback for excluded sites

8 sites (80 − 32 = 48 excluded from async) fall back to exclusion rather than a different pair. The 32 eligible sites are sufficient for a 70/15/15 site-level train/val/test split:
- 22 train, 5 validation, 5 test (roughly)

This is smaller than the originally planned 56/12/12. The plan must be updated.

---

## Section 7 (addendum): Split feasibility check — Strategy B confirmed

Analysis date: 2026-10-06
Source script: `src/training/split_feasibility.py`

### Common eligible set (all three models)

The synchronous baseline is the most restrictive model because it additionally requires S2 at T2. Using the common set ensures all three models are evaluated on identical sites.

| Category | Sites |
|---|---|
| Async eligible (bld+S1+S2@T1, bld+S1@T2) | 32 |
| Sync eligible — common set for all models | **28** |
| Async-only (S2@T2 missing, excluded from common) | 4 |
| Zero-positive-pixel sites | **0** |
| Usable sites for split | **28** |

The 4 async-only sites (`L15-0361E`, `L15-1479E`, `L15-1481E`, `L15-1538E`) lack S2 at December 2019. They are excluded from the common set to keep the comparison fair across all three models.

### Label validity — all 28 common sites confirmed valid

| Statistic | Value |
|---|---|
| Sites with zero positive pixels | 0 |
| Change fraction (min) | 0.044% |
| Change fraction (max) | 4.225% |
| Change fraction (mean) | 1.005% |
| Change fraction (median) | 0.733% |

No sites were excluded for label issues.

### Reproducible split (seed 42)

| Split | Sites | Role |
|---|---|---|
| Train | 20 | 71% |
| Validation | 4 | 14% |
| Test | 4 | 14% |

Split written to `experiments/scenario_2/configs/split_definition.json`. No overlaps confirmed.

### pos_weight from training sites only

| Metric | Value |
|---|---|
| Training sites | 20 |
| Total training pixels | 4,768,735 |
| Change pixels | 50,387 (1.057%) |
| No-change pixels | 4,718,348 |
| **pos_weight** | **93.6** |

Round to **94** for use in BCEWithLogitsLoss.

### Test set adequacy assessment

| Metric | SEN12-MT (Strat B) | OSCD (reference) |
|---|---|---|
| Test sites | 4 | 10 |
| Test pixels | 954,528 | 3,078,936 |
| Change pixels in test | 6,859 (0.719%) | ~159,077 |
| Relative volume | **31% of OSCD** | 100% |

**The test set is small.** 4 sites and ~955K pixels represents 31% of the OSCD test volume. Conclusions drawn from this test set should be treated as indicative rather than definitive. The paper must disclose this limitation.

### Strategy B vs Strategy C final comparison

| Metric | Strategy B (2018-03→2019-12) | Strategy C (earliest/latest) |
|---|---|---|
| Temporal gap | 21 months (uniform) | 12–26 months (variable) |
| Common eligible sites | 28 | ~51 |
| Train sites | 20 | ~36 |
| Val sites | 4 | ~8 |
| Test sites | 4 | ~8 |
| Test pixel volume vs OSCD | 31% | ~56% |
| Temporal confound | None | Variable gap (confound) |
| pos_weight | 93.6 | TBD (~89–100) |

**Strategy C gives ~1.8× more training data and ~2× larger test set with no zero-positive sites expected.** The cost is a variable 12–26 month temporal gap, which must be disclosed in the paper but does not invalidate the architecture comparison (all three models see the same variable-gap pairs).

### Final recommendation

**Proceed with Strategy B (2018-03 → 2019-12) as specified.** The 28-site common set is defensible. The small test set (4 sites) is a genuine limitation that must be disclosed. If during training the results are statistically inconclusive due to test-set size, upgrading to Strategy C is a legitimate post-hoc option — but only if the test split is re-locked before any test evaluation.

The split is now locked in `experiments/scenario_2/configs/split_definition.json` and must not be modified after training begins.


| Split | Sites | Role |
|---|---|---|
| Train | 22 | 70% of 32 |
| Validation | 5 | 15% |
| Test | 5 | 15% |

This is smaller than the original 56/12/12 plan. The reduction is due to buildings label unavailability at fixed endpoints. The paper should clearly state that 32 of 80 sites have complete multimodal data at the chosen endpoints.

However: **Strategy C (earliest/latest) gives 59 async sites**, which would allow a closer-to-planned 41/9/9 split. If the variable-interval confound is acceptable to you (and can be controlled for in the paper), Strategy C provides substantially more training data.

**This is the key scientific decision that requires your input:**
- **Strategy B (2018-03→2019-12): 32 sites, uniform 21-month interval, cleaner comparison.**
- **Strategy C (earliest/latest): 59 sites, variable 12–26 month interval, more data.**

Both are scientifically defensible with appropriate disclosures.

---

## 8. Unresolved data quality issues

1. **Zero-positive sites**: None found at threshold 0.5 among the 27 Strategy A sites checked. This is a positive result — no sites need exclusion on this basis.

2. **Very low change fraction** (< 0.1%): `L15-0434E-1218N_1736_3318_13` was noted at 0.05% in the previous inspection. This site may contribute near-zero signal. It is not excluded but should be flagged in per-site results.

3. **S2 bad data rate**: 395 bad S2 timestep indices across 80 sites. The concentration of bad S2 observations in certain sites (e.g. `L15-0571E-1075N_2287_3888_13` has 17/20 bad S2) primarily drives the exclusion from fixed-pair strategies.

4. **pos_weight**: The 89.0 estimate is from 27 Strategy A sites. The final value must be recomputed from whichever 22 sites land in the training split. Do not use 89 as a hard prior without recomputing.
