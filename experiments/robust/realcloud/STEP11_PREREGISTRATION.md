# Step 11 pre-registration: real cloudy post-change Sentinel-2 scenes (evaluation only, no training)

Written 2026-11-11, before any change model was evaluated on a real scene except the smoke test of §7. Base commit
`4c721d8` (branch `main`). This file and the files in §8 are frozen by SHA-256, and the file is committed before the
evaluation runs; `run_real_cloud.sh` refuses to start otherwise. Any later change is reported as a deviation.

## 1. Why, and what is already known

Every cloud result in §5 of the manuscript uses synthetic or composite clouds (Stage 3 review, R3 W2; Tier 2 item 13).
Step 11 evaluates the frozen models on real partly cloudy acquisitions near OSCD date 2.

**Known before writing (no change model involved):**

- Scene metadata and city-box cloud fractions (`experiments/robust/feasibility/real_cloud_gee.json`, Earth Engine
  COPERNICUS/S2_CLOUD_PROBABILITY ≥ 40 over the city box, ±30 days of OSCD date 2).
- The selected scenes (§3) and a data check on them: each control matches OSCD's own date-2 image (B4 mean equal to
  three decimals; median band correlation 0.86–0.96, Saclay 0.71). The Las Vegas "overcast" scene (cloud probability
  0.96) correlates 0.91 with the clear scene and its B4 mean barely rises, so it is probably bright desert or thin cloud
  flagged by the detector, not opaque cloud; Dubai's cloudy scene (0.15) also correlates 0.93. Both are kept as the fixed
  rule selected them.
- s2cloudless outputs on every scene, run before this file as Step 5 ran its detector before its test evaluation. The
  masked fraction of each scene at Step 5's frozen operating points (bal / rb95 / rb99): on the 7 cloudy scenes
  0.14–0.78 / 0.67–1.00 / 1.00; on the 3 overcast scenes 1.00 at all three; on the 10 clear controls 0.00–0.19 /
  0.10–0.98 / 0.91–1.00. **rb99 masks essentially the whole image everywhere, so it is reported descriptively only, and
  rb95 also discards most of many clear scenes.**
- Synthetic and composite results of Phases 3–5 and the exploratory SAR-only comparison of manuscript §5.8.

## 2. Hypotheses (two-sided)

On the 7 partly cloudy test cities, at the frozen clean thresholds (§4):

- **H1:** does zero-filling the s2cloudless rb95 mask change B's F1 against the unmasked real scene?
- **H2:** does B with the rb95 mask differ from the SAR-only U-Net fallback?
- **H3:** without a mask, does B (modality dropout) differ from A (full-input training)?

## 3. Data and conditions

- **Scenes** (`download_real_cloud.py`, rule fixed before download, applied to the stored cloud fractions): per test
  city, the full-coverage day within ±30 days of OSCD date 2 with city-box cloud in [0.05, 0.70] nearest to date 2
  (ties: earlier); if none, the nearest day with cloud ≥ 0.90 ("overcast"); and OSCD's own date-2 day as a control.
  COPERNICUS/S2_HARMONIZED L1C, 13 bands, same-day tiles mosaicked, on the city's first Sentinel-1 orbit grid, resized to
  the label grid (bilinear), /10000, clipped to [0, 1], as Step 8.

| city | real scene | day (Δ from date 2) | city-box cloud |
|---|---|---|---|
| brasilia | cloudy | 2017-10-25 (+8) | 0.391 |
| norcia | cloudy | 2017-10-08 (−10) | 0.354 |
| saclay_w | cloudy | 2017-09-01 (+3) | 0.676 |
| valencia | cloudy | 2017-10-23 (−15) | 0.430 |
| dubai | cloudy | 2018-03-25 (−5) | 0.152 |
| milano | cloudy | 2018-01-17 (−5) | 0.450 |
| chongqing | cloudy | 2018-04-07 (+5) | 0.084 |
| montpellier | overcast | 2017-11-04 (+5) | 1.000 |
| rio | overcast | 2017-10-16 (+5) | 1.000 |
| lasvegas | overcast | 2018-02-10 (+5) | 0.956 |

- **City sets:** primary = the 7 cloudy cities; overcast = montpellier, rio, lasvegas; all = the 10 test cities.
- **Conditions** (only the post-change optical image changes; SAR and the pre-change image are as in every step):
  `oscd` (OSCD's own image; anchor), `control` (same-day Earth Engine scene), `real` (no mask), `real_bal`,
  `real_rb95`, `real_rb99` (s2cloudless 1.7.3 probability, Step 5's frozen operating points t / dilation r: bal 0.65 / 6,
  rb95 0.25 / 8, rb99 0.05 / 16, zero-filled as Step 5), `control_rb95`, `missing` (post-change optical zero).
- **Models:** A `p2_full`, B `p2_dropout` (DS-UNet), SAR-only `p6_unet_sar`, seeds 1–10, read-only, hashed before and
  after. **Thresholds:** each checkpoint's frozen clean-validation (`full`) threshold, as Phases 3–5 and §5.8; nothing is
  chosen on test data.

## 4. Evaluation and analysis

**Evaluation (`eval_real_cloud.py`, once).** Per seed and city, (101, 4) TP, FP, FN, TN per condition. **Anchors**
(any failure: nothing saved, the run stops): A and B `oscd` pooled over the 10 cities equal Step 5's stored
`test__clean` counts at all 101 thresholds; SAR-only pooled equals its Step 6 counts at the selected threshold and at
0.5; SAR-only gives identical counts with the OSCD and the real post-change image. Prints no metric.

**Analysis (`analyze_real_cloud.py`).** Per seed, counts pooled over the cities of a set, F1 from pooled counts; paired
over seeds 1–10; paired t-test, two-sided, Holm within each family; mean Δ, 95% CI, d_z, seeds with Δ > 0, Wilcoxon
alongside. Labels "higher" / "lower" if Holm p < 0.05 with that sign, else "no detectable difference". No equivalence
margin.

- **Primary (7 cloudy cities, Holm over 3):** H1 B `real_rb95` − B `real`; H2 B `real_rb95` − SAR-only; H3 B `real` − A
  `real`.
- **Secondary (7 cloudy cities, Holm over 4):** B `real_bal` − B `real`; B `real_rb95` − B `missing` (masking against
  discarding the image); A `real` − SAR-only; B `control` − B `oscd` (processing-chain check).
- **Overcast (3 cities, Holm over 3)** and **all 10 cities (Holm over 3):** the three primary contrasts.
- **Descriptive, no tests:** F1, precision, recall and false-alarm rate of every arm and condition for each set
  (including `real_rb99` and `control_rb95`); per-city F1; for the primary contrasts, the pooled Δ with each cloudy city
  left out (exploratory).
- **Sensitivity:** with 10 pairs and power 0.8, the minimum detectable effect is 1.22 SD of the paired differences at
  α = 0.05/3 (first Holm step).

## 5. How the result will be read (pre-specified)

- **H1 higher:** on real clouds, zero-filling detected cloud helps the dropout model, as on composites (Step 5). Because
  rb95 masks most of several images, the secondary `real_rb95` − `missing` contrast says whether the mask beats simply
  discarding the image; if it does not, the gain is read as "discarding the image", not "locating cloud".
- **H2:** "higher" supports, on real scenes, the §5.8 observation that B with a recall-biased mask scores above a
  SAR-only fallback; "no detectable difference" or "lower" means that observation does not carry over.
- **H3 higher:** dropout training reduces the damage of real cloud, as for synthetic cloud (Phase 3).
- Results concern 7 (or 3, or 10) cities with one real scene each, so seeds do not replicate scenes; the scenes lie 3–15
  days from the label date, so a little change may fall between them; cloud cover differs by city and is not controlled.
  Every result is reported whatever its direction.

## 6. Stopping rules and failures

`run_real_cloud.sh` stops if this file is not committed or has changed since its commit, if the smoke log does not end
with `n_failed 0`, if any hash below has changed, if an anchor fails or if the analysis fails. No seed or city is
excluded. Both scripts refuse to overwrite.

## 7. Smoke test

`smoke_real_cloud.py` → `smoke_real_cloud.log`: **7 checks, 0 failed**. The evaluation ran on seed 1 of A, B and
SAR-only into a temporary folder that was deleted afterwards; only its exit status, anchors and array keys were read
(no metric was computed or printed from it). All anchors passed, including the exact reproduction of Step 5's stored
clean test counts for A and B. The analysis ran on synthetic random counts with the same layout; a contrast between two
identical conditions was exactly zero; both scripts refuse to overwrite. The smoke test was run twice (the second run
wrote the log); both passed.

## 8. Frozen files (SHA-256)

| file | SHA-256 |
|---|---|
| `experiments/robust/realcloud/download_real_cloud.py` | `483dc78c5706ec43d0c8f02dd9635db1befbea6e8dc9f23b1371500af2b5166e` |
| `experiments/robust/realcloud/export_real_cloud_inputs.py` | `5a54975788dcc5df25309e5cdb47dd36019446a06877d0dfec5df55515748f5d` |
| `experiments/robust/realcloud/detect_real_cloud.py` | `3c77dcd48d783580d2910f61ef4c657c459a315be0c19b26c934af2acf21f77b` |
| `experiments/robust/realcloud/eval_real_cloud.py` | `489ef33513d6649738a2780398bb1d4a16d31e7e79c12a196cf303ae7b344480` |
| `experiments/robust/realcloud/analyze_real_cloud.py` | `4949bd8fa83e15c8fb2901d9c14231ec6f1d78faad740d37605b595ad2575c1d` |
| `experiments/robust/realcloud/smoke_real_cloud.py` | `dd169aba2728d400fbd1cde9229e3783f7edf936b8a35f49c8dbc3a84ce50802` |
| `experiments/robust/realcloud/smoke_real_cloud.log` | `8e2b574e575a14ca9429a5fb9a37fd2718b6dac8b25131d42d81e3ebbe007e9f` |
| `experiments/robust/realcloud/run_real_cloud.sh` | `b0ab7ff2c6d8dcc5b8a2848093be68f4543515caf4302837ce22258b8f422e36` |
| `experiments/robust/feasibility/real_cloud_gee.json` | `9b2175d0135e01cf7598db3f191e2e5b71a45a2e143958a0834c22b38d951bb3` |
| `experiments/robust/phase5/operating_points.json` | `ae0ec54611d0a076b8f6474fd1bf088cdc51b4127ddfb812273542458d582619` |
| `datasets/oscd_s2_realcloud/manifest.json` | `0527b1a853e57d86e09ec50e5b8598465e1d4fa99d450f6ffc49d760579881cf` |
| `experiments/robust/realcloud/detector_outputs/detector_info.json` | `bfd91d2939624136676597212a8c64d8c32a337657997d4399bdf225bd04bbe8` |
| `eval_phase4a.py` | `5d831a81126b97c99778fa835ec35ef068520b30b960756c3c3b97b3ccc70f62` |
| `src/phase5/composite.py` | `8547387921807bde16d11019e2f2b64e208895a6d9a8a9cb1d2e3137ff89d61d` |
| `src/training/train_robust.py` | `c7d812e3fd8dcf2c9bd8c7f4d3faeb1bee0c97fc256f83a76ee2194e15a6436c` |
| `src/training/train_rrd.py` | `cf918eb7481f8cd98352a4702f16913f03f373cb9d5cd077d9fe6f4b0adcbb32` |
| `src/datasets/oscd_dataset.py` | `bc9b64bc2fb9808efb2d8b6767a594f19385e52f4514c09633b3a7af8486fc43` |
| `src/evaluation/metrics.py` | `6d49b1892b9a2689bc8c4c23673535b06e5dc4f45df86c1bd3c0279a9f7b4854` |
| `analyze_phase3.py` | `96bd2cc6811321ad00438f45cb81d3e01cc485d4406c2aa5fc7e770a6354a04b` |
| `experiments/robust/step9/analyze_step9.py` | `69dcf48302271bccd6c502702ff4580afcb3f140e71343383abca65905cc2b33` |
| `verify_prereg_hashes.py` | `7455f73c1ea05f75673583d8b939ea9af38616685c42a5eaefe372b0e85c1c4d` |
| `datasets/oscd_s2_realcloud/*.tif (20, concatenated, sorted)` | `53a43405c5afa814dc8f893ecd8b948d225fc96a6a593b2adbcc392a4a5d0668` |
| `experiments/robust/realcloud/detector_outputs/*.npz (20, concatenated, sorted)` | `32803d9b03fbb0a162ae4695fe7167a8504de7ce8bfd6b4e40d027ed509228f2` |
| `checkpoints/robust/p2_full_seed*.pt (10, concatenated, sorted)` | `5817ec427596daa2062c84902eed54e727681457e33aae4d4129b59c504b844e` |
| `checkpoints/robust/p2_dropout_seed*.pt (10, concatenated, sorted)` | `8bc2646ed372b5e1c60f9a0d17f20b7735064d94d54d284bbed785e7a13f40e2` |
| `checkpoints/robust/p6_unet_sar_seed*.pt (10, concatenated, sorted)` | `5fd758a336d0edc85eac387912a3a668e7f706384af2e7dee675213c5d75b03a` |
| `experiments/robust/phase5/runs/p2_*_seed*.npz (20, concatenated, sorted)` | `37d3287ac7a7e8fa2d8f630c294ee17c911e22a9d99ff3a1e51322138a50d423` |
