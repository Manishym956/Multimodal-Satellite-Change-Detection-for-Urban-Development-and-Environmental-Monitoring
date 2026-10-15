# RQ2 Decision Memo: Temporal-Gap Experiment Readiness

**Decision: Do not begin the temporal-gap training sweep yet.** The eligibility inventory is reproducible under the repository's current flag conventions, but provenance for several record-level quality flags is not authoritative. The all-gap common-site design is also too small for a defensible controlled comparison. No model training or untrained-gap metrics were run for this memo.

## Research question

RQ2 asks whether historical Sentinel-2 optical imagery improves Sentinel-1 SAR-based urban change detection when synchronous optical imagery is unavailable, and how performance changes as the historical-optical-to-target-SAR gap varies. The existing March 2018 optical / December 2019 SAR interval has a **21-calendar-month gap**. It is one trained interval, not a temporal-gap sensitivity experiment.

## What the evidence establishes

- **Audit result:** 644 locked-site/month rows across 28 sites and 23 candidate months: 515 confirmed eligible, 84 ambiguous, and 45 ineligible under the current audit rules. All 85 optical files that conflict with usability metadata are recorded individually in `eligibility_verification.json` and `optical_disagreement_files.csv`.
- **Upstream provenance:** the [Zenodo SEN12 Multi-Temporal Urban Mapping release](https://zenodo.org/records/7794693) documents monthly mean Sentinel-1, cloud-free Sentinel-2, and monthly built-up labels. It lists `sen12_mt_urbanmapping.tar.gz` and published MD5 `5d34fc4537c7974c31824baaaea050f5`. This record does not define the local `s2`, `masked`, or bad-index semantics.
- **Repository implementation:** `src/training/generate_manifest.py` maps zero-based metadata list positions to bad-S2 indices and builds raster paths from record year/month. The check matched index membership and filename dates for all 1,889 generated manifest rows. `src/datasets/scenario2_dataset.py` reads the fixed experiment dates directly without consulting these flags.
- **Unresolved provenance:** no original dataset README/data dictionary, flag-generation code, record-level quality rationale, source archive, or trusted local per-file checksum manifest was found in the project. Therefore, a present/readable TIFF cannot overrule a bad/masked flag, and a flag cannot establish the scientific defect by itself.
- **March reference:** March 2018 is complete at all 28 locked sites (20 train, 4 validation, 4 test). Locked split SHA-256 remains `5d0841adf0c651166cb18ecea310c6ef8cce69172065fc2d699a29083c272e27`.

## Candidate-gap coverage

Gap is defined as calendar-month index difference from the candidate historical optical month to target SAR month 2019-12. Counts below are confirmed eligible sites under the current audit rules, shown as total / train / validation / test; ambiguous and ineligible counts are out of 28.

| Optical month | Gap (months) | Confirmed sites: total / train / val / test | Ambiguous | Ineligible |
|---|---:|---:|---:|---:|
| 2018-01 | 23 | 21 / 15 / 4 / 2 | 3 | 4 |
| 2018-02 | 22 | 22 / 15 / 4 / 3 | 5 | 1 |
| 2018-04 | 20 | 26 / 19 / 4 / 3 | 2 | 0 |
| 2018-05 | 19 | 23 / 16 / 3 / 4 | 4 | 1 |
| 2018-06 | 18 | 22 / 16 / 4 / 2 | 6 | 0 |
| 2018-07 | 17 | 18 / 13 / 3 / 2 | 8 | 2 |
| 2018-08 | 16 | 18 / 13 / 3 / 2 | 6 | 4 |
| 2018-09 | 15 | 17 / 14 / 1 / 2 | 10 | 1 |
| 2018-10 | 14 | 24 / 18 / 2 / 4 | 3 | 1 |
| 2018-11 | 13 | 24 / 17 / 3 / 4 | 3 | 1 |
| 2018-12 | 12 | 24 / 18 / 3 / 3 | 2 | 2 |
| 2019-01 | 11 | 23 / 17 / 3 / 3 | 3 | 2 |
| 2019-02 | 10 | 17 / 14 / 2 / 1 | 5 | 6 |
| 2019-03 | 9 | 21 / 15 / 3 / 3 | 1 | 6 |
| 2019-04 | 8 | 23 / 16 / 4 / 3 | 0 | 5 |
| 2019-05 | 7 | 24 / 16 / 4 / 4 | 3 | 1 |
| 2019-06 | 6 | 23 / 17 / 3 / 3 | 3 | 2 |
| 2019-07 | 5 | 21 / 16 / 2 / 3 | 5 | 2 |
| 2019-08 | 4 | 23 / 17 / 3 / 3 | 3 | 2 |
| 2019-09 | 3 | 22 / 17 / 3 / 2 | 4 | 2 |
| 2019-10 | 2 | 26 / 19 / 3 / 4 | 2 | 0 |
| 2019-11 | 1 | 25 / 18 / 3 / 4 | 3 | 0 |

Coverage ranges from **17 to 26 confirmed sites** across untrained gaps. The existing 2018-03 interval has 28/28 but is not an untrained gap candidate. See `eligibility_verification.json` for the full per-site list and split breakdown at each month.

## Fixed-site limitation and design decision

The intersection of sites confirmed eligible at every one of the 22 untrained candidate gaps is only **7 sites: 5 train, 1 validation, 1 test**. This does not support a useful locked-split, all-gap comparison; in particular, one test site makes test performance highly site-dependent. Training each gap on its variable eligible sites would confound temporal gap with changes in site composition.

After provenance is resolved, either (a) preselect a smaller set of gaps with an adequately sized common site intersection, without changing the locked split, or (b) use variable-site results only as explicitly composition-confounded exploratory evidence and report site membership for every gap. Do not present either as a controlled temporal-gap effect unless site composition is held fixed or its effect is addressed in a predeclared design.

## Exact evidence required to unblock training

1. **Authoritative field definitions:** a dataset data dictionary, release documentation, or written author/processor confirmation defining `metadata.json.s2`, `metadata.json.masked`, and `bad_data.json.S2`—including whether false/bad/masked means absent, cloudy, invalid, incomplete coverage, processing failure, or another condition.
2. **Index provenance:** original code or documentation showing how bad-S2 indices were generated and their base/order (zero- or one-based, and which metadata list ordering was used). Repository implementation confirms its own zero-based convention but not the producer's convention.
3. **The 85 record dispositions:** source quality/provenance for each conflicting site-month, or an authoritative rule that resolves all three observed flag combinations. File existence, successful raster opening, and plausible pixel values are not sufficient.
4. **Source integrity:** retrieve the exact upstream archive and verify its published MD5, then establish that the local extraction matches it; alternatively obtain trusted per-file checksums and an original extraction manifest. Local metadata and bad-data SHA-256 fingerprints alone do not authenticate the rasters.
5. **Pre-training design lock:** once eligibility is resolved, document the chosen gap set, a fixed site membership strategy (or explicitly exploratory variable-site protocol), and the per-split sample counts. Preserve the current split definition and hash.

Once items 1–4 are available, rerun the read-only provenance/eligibility checks and record dispositions without editing the existing audit. Training can be considered only after item 5 is fixed and the site-composition limitation is addressed.

## Evidence files

- `experiments/scenario_2_temporal_gap/eligibility_verification.json`
- `experiments/scenario_2_temporal_gap/eligibility_verification.csv`
- `experiments/scenario_2_temporal_gap/fixed_site_intersection.json`
- `experiments/scenario_2_temporal_gap/provenance_investigation.json`
- `experiments/scenario_2_temporal_gap/provenance_investigation.md`
- `experiments/scenario_2_temporal_gap/optical_disagreement_files.csv`
- `docs/SCENARIO_2_DATASET_INSPECTION.md`
- `docs/SCENARIO_2_DATASET_MANIFEST.csv`
- `src/training/generate_manifest.py`
- `src/training/temporal_pairing_analysis.py`
- `src/datasets/scenario2_dataset.py`
