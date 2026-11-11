# Stage 4B pre-registration: model B with and without the historical optical image (RQ1 and RQ5 attribution; evaluation only)

Written 2026-11-02, after the Stage 3 simulated peer review and the Stage 4 exploratory re-analysis, and before
`eval_stage4b.py` ran on any test city. Base commit `dfeba6c` (branch `main`); the Stage 4 working-tree changes are
uncommitted. This file, `eval_stage4b.py` and `analyze_stage4b.py` are frozen by the SHA-256 hashes in §8, and the file is
committed before the evaluation runs. Any later change is reported as a deviation.

## 1. Why, and what is already known

Stage 4 left two attribution questions open (`paper/review/stage4/RESPONSE_MATRIX.md`, gap analysis):

- **RQ1.** At B's frozen clean-input threshold (the rule §5 uses), B without the post-change image scores +0.060 F1 above
  the SAR-only U-Net (exploratory). That is a difference between two networks. The same-model contrast (B with against
  without the historical image) exists only at each input's own validation threshold and at 0.5, so it is unknown how
  much of the +0.060 comes from the historical image.
- **RQ5.** Step 8 found a small false-alarm reduction in shrubland for B against the SAR-only U-Net. No count of B without
  the historical image by land-cover class exists, so that reduction cannot be attributed to the image.

**Seen before writing this file (partial unblinding, stated openly).** For B without any optical input (`no_opt`), the
test counts pooled over the 10 test cities are stored at its own validation threshold and at 0.5 (Phase 2
`metrics.json`). The within-B contrast is therefore known at those thresholds: present − absent −0.010 (95% CI
[−0.026, +0.007]) and −0.003 ([−0.025, +0.019]). Not seen: any `no_opt` count at another threshold, any per-city `no_opt`
count, and any `no_opt` count by land-cover class. A validation-city smoke test (beihai, seed 1, nothing saved) checked
the code path only. Stage 4B is therefore a follow-up evaluation of models, data and contrasts that have already been
analysed at other thresholds, not an independent study or a test of previously unseen hypotheses.

## 2. Questions and hypotheses (two-sided)

- **H1 (primary, RQ1 attribution):** at B's frozen clean threshold, does the historical image change B's test F1?
- **H2 (secondary):** the same at B's frozen `no_o2` threshold, and for precision and recall at the clean threshold.
- **H3 (RQ5 attribution):** within B, does the historical image change the false-alarm rate (and recall) in any of the 7
  WorldCover classes tested in Step 8?

## 3. Data, models and inputs (no training)

- **Test cities:** the 10 OSCD test cities (brasilia, montpellier, norcia, rio, saclay_w, valencia, dubai, lasvegas,
  milano, chongqing). RQ5 uses Step 8's 9 primary cities (all but chongqing), as registered for Step 8.
- **Inputs:** Step 8's pipeline unchanged (`rrd.RRDDataset("test", "hafner_vv")`, DS-UNet VV rasters, first listed orbit
  per city, cuDNN deterministic). Post-change optical image absent (zero) in both conditions.
  - *present* (`no_o2`): OSCD's own date-1 Sentinel-2 image in the pre-change optical slot (Step 8 tag `oscd`).
  - *absent* (`no_opt`): both optical slots zero; SAR t1 and t2 only.
- **Checkpoints:** B (`p2_dropout`, modality dropout), seeds 1–10, `checkpoints/robust/p2_dropout_seed{s}.pt`, each the
  step selected on validation cities in Phase 2. Loaded read-only; their SHA-256 hashes (§8) are recorded before
  loading and checked again after the last seed. Not retrained, fine-tuned or modified.
- **Thresholds (all frozen in the checkpoints; none chosen on test data):** `tau_clean` = the checkpoint's `full`
  threshold (equal to Phase 3's frozen clean threshold for every seed, checked from stored files); the `no_o2`
  threshold; 0.5 only in anchors.
- **Land cover:** ESA WorldCover v100 (2020), nearest neighbour to the label grid, as Step 8.

## 4. Evaluation (`eval_stage4b.py`, once; outputs in `experiments/robust/stage4b/runs/`)

Saved per seed and city: (101, 4) TP, FP, FN, TN for each condition, and (101, 4) counts per WorldCover class.
**Anchors** (any failure: nothing saved, the run stops and the failure is reported):

1. *present* equals Step 8's stored `<city>__oscd` counts at all 101 thresholds, for every city and seed;
2. *present* class counts at the `no_o2` threshold equal Step 8's stored class counts;
3. class counts sum to the city counts at every threshold, for both conditions;
4. *absent* pooled over the 10 cities equals the stored Phase 2 `no_opt` test counts at its own threshold and at 0.5;
5. checkpoint hashes unchanged after the run.

The script refuses to overwrite and prints no test metric. Existing result files are read, never written.

## 5. Analysis (`analyze_stage4b.py`)

Per seed, counts are pooled over cities and the metric is computed from the pooled counts. Contrasts are present − absent,
paired over the 10 seeds: paired t-test, two-sided, α = 0.05, with 95% CI, d_z and an exact Wilcoxon test alongside.

- **Primary:** F1, 10 test cities, both conditions at `tau_clean`. Label "higher with the image" / "lower with the image"
  if p < 0.05, else "no detectable difference".
- **No equivalence test.** No smallest effect of scientific interest in F1 is established for this task, so no
  equivalence margin is set and no result is called negligible or equivalent. The primary result is the difference
  test with its 95% CI; a null is not evidence of no effect, and the CI bounds are what the paper reports as the range
  of contributions compatible with the data.
- **Secondary (Holm over 3):** F1 at the `no_o2` threshold; precision and recall at `tau_clean`.
- **RQ5 (Holm over 7, own family):** false-alarm rate per class (tree cover, shrubland, grassland, cropland, built-up,
  bare, water), 9 primary cities, both conditions at the `no_o2` threshold (Step 8's registered RQ5 operating point).
  **RQ5 recall:** the same, own Holm family over 7.
- **Descriptive, no tests:** (a) at `tau_clean`, (present − SAR-only) = (present − absent) + (absent − SAR-only), each with
  its paired CI; SAR-only is `p6_unet_sar` at its frozen threshold (Step 8 counts); (b) false-alarm rate per class at
  `tau_clean`; (c) F1 of each Step 8 image age (1–18 months) minus *absent*, 9 primary cities, `no_o2` threshold.
- **City level (exploratory, primary contrast only):** the method as corrected after a separate check of Stage 4 by a fresh
  session of the same AI assistant (2026-11-02; not independent in error process). It reports standard errors of the
  same pixel-pooled estimate with seeds resampled and with cities left out (jackknife), a paired two-stage percentile bootstrap (10000 replicates, RNG seed 20261103), a jackknife-t interval,
  the range with one city left out, and unweighted per-city Δ with each city's share of change pixels. With 10 cities of
  unequal size these intervals are approximate and are not used for any decision.

## 6. How the result will be read

- **H1 positive:** some of B's clean-threshold margin over the SAR-only U-Net comes from the historical image; the
  decomposition shows how much.
- **H1 null:** no contribution of the image is detected; the paper reports the 95% CI as the range of contributions
  compatible with the data and does not claim the contribution is zero.
- **H1 negative:** the image lowers B's F1 at this threshold.
- **H3:** a class with a Holm-detectable within-B difference is attributable to the image. If the shrubland reduction is
  not detectable within B, the paper keeps the Step 8 result as a between-model difference only.
- **Text:** every result is reported, whatever its direction. The registered Step 6 and Step 8 results remain primary
  and unchanged. Stage 4B is labelled as a post-review evaluation with the partial unblinding stated in §1.

## 7. Change before freezing (2026-11-02, before any Stage 4B test evaluation)

The draft set an equivalence margin of ±0.03 F1 (TOST on the primary contrast). Its stated basis was the effect
detectable with 80% power for n = 10 and the SD of already-observed within-B differences; that is statistical
convenience informed by seen results, not a scientific smallest effect of interest. The equivalence test was removed
from this file and from `analyze_stage4b.py` (function `tost`, constant `MARGIN`, report line). Nothing else changed.

## 8. Frozen files (SHA-256)

Filled in when frozen; the file is committed with these values before the evaluation runs.

| file | SHA-256 |
|---|---|
| `experiments/robust/stage4b/eval_stage4b.py` | `2b82687b68c18a6d8255ad54c6a5a6ae513897d2a0b4d8651010416ed4eb1746` |
| `experiments/robust/stage4b/analyze_stage4b.py` | `c16dcbb666aeeaa016accd5fe4c95b6161f9f2d63e9bc09dbdc7b1386d50cbde` |
| `checkpoints/robust/p2_dropout_seed1.pt` … `seed10.pt` | listed in `checkpoint_sha256.txt` (same folder) |

**Expected cost:** about 5 minutes on the RTX 4060 laptop GPU (20 forward passes per seed), seconds for the analysis.
