# Step 8 pre-registration: historical optical image age and land cover (RRD RQ1, RQ2, RQ5; evaluation only)

Written 2026-10-23 before `eval_phase8.py` ran on any test city. Base commit `2c74a73`, branch `robust-phase4`. Plan:
`docs/final/FINAL_RESEARCH_PLAN.md`. Any later change is reported as a deviation.

**Timing, stated openly.** `eval_phase8.py`, `analyze_phase8.py` and `audit_p8.py` were written at 19:25–19:30 IST and
not changed since (file times); Step 6's results (finished 22:45) were read before this file was written. Step 6 found no
detectable difference between SAR + OSCD's own historical image and SAR alone (RQ1); Step 8's design and contrasts were
fixed in those scripts before that result existed.

## 1. Questions and hypotheses

- **RQ2 (primary):** for B (modality dropout) without the post-change optical image, does test F1 change with the age of
  the historical optical image (1, 3, 6, 12, 18 months before OSCD date 2)? Two-sided.
- **RQ1 by gap (secondary):** at each age, does B (and the asynchronous model) differ from SAR alone?
- **RQ5 (secondary):** in which WorldCover land-cover classes does B with the historical image raise or lower the
  false-alarm rate relative to SAR alone?
- Known when writing: Song et al. (2026, §5.4) report F1 falling from 83.99% (1 month) to 78.31% (12 months) for
  construction-ready bare land in Guangzhou. Step 6 (above).

## 2. Data (downloaded and checked before this file)

- **Historical images** (`download_gee.py s2`, Earth Engine COPERNICUS/S2_HARMONIZED, L1C, 13 bands): for each city,
  OSCD's own date-1 acquisition day (`gee_t1`, same-date control) and one image per target date 2 − g, g ∈ {1, 3, 6,
  12, 18} months. Rule (fixed before any model saw a scene): within ±45 days, full coverage and a cloud-probability
  image required; the day nearest the target with cloud fraction ≤ 0.02 over the city box (s2cloudless probability ≥ 40),
  else the clearest day. Exported on the DS-UNet SAR grid (bands requested in groups of 4 to respect the request-size
  limit; L1C integer DN, uint16 or float32). Scene list: `phase8/s2_scene_manifest.json` (144 rows, every city × tag has
  a scene; OSCD's own day found for 24/24 cities; median 7 and maximum 45 days from the target).
- **Primary city set** (from the manifest only): test cities whose scene has cloud fraction ≤ 0.10 at every gap:
  brasilia, montpellier, norcia, rio, saclay_w, valencia, dubai, lasvegas, milano (9). Chongqing is excluded (cloud
  0.977 at 3 months, 0.214 at 18 months) and enters only the all-city sensitivity analysis. This meets the no-go
  threshold (≥ 8) set in `docs/final/DATA_FEASIBILITY_REPORT.md`.
- **Alignment control (smoke test, data only):** the Earth Engine date-1 image, processed as the evaluation does
  (bilinear resize to the label grid, /10000, clip), against OSCD's own date-1 bands over all 24 cities: median band
  correlation 0.940, median phase-correlation shift 0.27 px, every city below 1 px (`phase8/smoke/alignment.json`).
- **Land cover:** ESA WorldCover v100 (2020), nearest neighbour to the label grid. It post-dates OSCD date 2, so
  "built-up" includes new construction; it is a stratification, not a label.

## 3. Models and evaluation (`eval_phase8.py`, once per checkpoint; no training)

Frozen checkpoints and thresholds, seeds 1–10: B `p2_dropout`, async `p6_async`, A `p2_full` (DS-UNet, setting `no_o2`:
SAR t1 and t2, the historical image in the pre-change optical slot, post-change optical zero, each run's frozen `no_o2`
threshold), and the SAR-only U-Net `p6_unet_sar` (reference, full input, its frozen threshold). Tags: `oscd` (OSCD's own
date-1 image: the anchor), `gee_t1`, `gap01`…`gap18`. Saved per city: (101, 4) counts per tag, counts per WorldCover class
at the frozen threshold, and for the DS-UNet arms the full-input pass.
**Anchors** (failure: run not saved): tag `oscd` pooled over the 10 test cities equals the stored `no_o2` counts (full
for the SAR-only arm and for the full-input pass) at the stored threshold and at 0.5; land-cover counts sum to the counts at
the frozen threshold. Prints no test metric.

## 4. Analysis (`analyze_phase8.py`, fixed by hash below)

Counts pooled over the primary cities. Paired over seeds; Holm within each family; Wilcoxon alongside.

- **primary_rq2:** B, per seed, the least-squares slope of F1 against g (months; 5 gaps); one-sample two-sided t-test of
  the 10 slopes against 0. Labels: "F1 falls with the gap" / "F1 rises with the gap" if p < 0.05, else "no detectable
  effect of the gap".
- **secondary_B_vs_sar (Holm over 5)**, **secondary_async_vs_sar (Holm over 5):** F1 at each gap − SAR-only.
- **secondary_async_slope:** as the primary, for async.
- **secondary_source (own family):** B@gee_t1 − B@oscd (processing chain, same date); B@oscd − SAR-only.
- **rq5_landcover (Holm over the classes with ≥ 1% of the pooled pixels):** false-alarm rate, B@oscd − SAR-only.
- **Descriptive:** every metric per arm and tag; the slope and F1 on all cities with a scene at every gap (sensitivity);
  F1 per tag split by whether the historical image precedes OSCD date 1; per-class false-alarm rate and recall.
- **Sensitivity:** minimum detectable effect 1.00 SD of the paired differences (slopes) at α = 0.05; 1.33 SD at α = 0.05/5.
- `audit_p8.py` (written before any Step 8 run) recomputes everything from the saved counts.

## 5. Interpretation guidance (pre-specified; not tests)

OSCD labels mark change between date 1 and date 2. Images taken 1–6 months before date 2 lie between the two dates and
may already show part of the labelled change; at 18 months about half precede date 1. The gap therefore changes both the
age of the context and how much of the change it already shows, and results are reported with that confound. A negative
slope agrees with Song et al.; a positive slope would mean recent images hurt (consistent with the confound); a null is
"no detectable effect of the image's age under this design". Results concern OSCD urban change, frozen models trained
with OSCD's own date-1 image, and zero post-change optical input.

## 6. Smoke test (no test output)

`smoke_phase8.py` → `phase8/smoke/smoke_results.json`, log `logs/robust/p8_smoke.log`: **8 checks, 0 failed** (manifest
complete; 9 primary cities; alignment as above; the evaluator on the 4 validation cities with `p2_dropout_seed1`
evaluates every tag and its land-cover counts sum; analysis and audit agree on synthetic counts).

## 7. Hashes (sha256)

| file | sha256 |
|---|---|
| `eval_phase8.py` | `eb8e5295183cbbe99601d89e946c4d315e2bcc2f63b6867b53c64f81b60a5453` |
| `eval_phase7.py` | `a17599a62035be979cffbf2c6cc9dc83b7b7368f3b516e593e12d150e13180a7` |
| `analyze_phase8.py` | `ae85ed7789c5967638de02c6db455100954c847039bebb548cac74eabfbe384a` |
| `analyze_phase7.py` | `0a9b402302fd4fab33265c1c13fdd26de4acbe104547d50352a8020bb2c555f4` |
| `analyze_phase3.py` | `96bd2cc6811321ad00438f45cb81d3e01cc485d4406c2aa5fc7e770a6354a04b` |
| `audit_p8.py` | `d21526759b42a44cefe641855bf691df85e76a64227eaa240724dfddc48d6245` |
| `smoke_phase8.py` | `1025a844c15a704ef44503863a5b3ded23e1d1c782c7b8459a77d5fc420068f1` |
| `src/training/train_rrd.py` | `cf918eb7481f8cd98352a4702f16913f03f373cb9d5cd077d9fe6f4b0adcbb32` |
| `src/training/train_robust.py` | `c7d812e3fd8dcf2c9bd8c7f4d3faeb1bee0c97fc256f83a76ee2194e15a6436c` |
| `src/datasets/oscd_dataset.py` | `bc9b64bc2fb9808efb2d8b6767a594f19385e52f4514c09633b3a7af8486fc43` |
| `verify_prereg_hashes.py` | `7455f73c1ea05f75673583d8b939ea9af38616685c42a5eaefe372b0e85c1c4d` |
| `logs/robust/p8_run.sh` | `b0228478013990c0489818d043d23addd4b080368da0fd906574e380e3e61114` |
| `logs/robust/p8_smoke.log` | `0383ca84652919fe20eafd584a2d3b9752266019da747c4852f1179ee150e719` |
| `phase8/smoke/smoke_results.json` | `ca8983b152ad0ddf65bc5aa534740a57457f939c1a661856f533e3cac5aef4ca` |
| `phase8/smoke/alignment.json` | `55663c8ce8d3293bfbf8ae88b114cb736ff456ef282c0a3f69934afb427930b5` |
| `phase8/s2_scene_manifest.json` | `e352a76502bfc4afc23ca2fe12a6be6a61a0b1c9a5d9811576caf60e6a91428e` |
| `../../datasets/oscd_s2_gee/manifest.json` | `e352a76502bfc4afc23ca2fe12a6be6a61a0b1c9a5d9811576caf60e6a91428e` |
| `../../datasets/oscd_s2_gee/*.tif (144, concatenated, sorted)` | `89c776e61e4bf9a8dd923bd7e51aec572cceb0e8208bc1b4ce27d5f7616b5e9e` |
| `../../datasets/oscd_wc_gee/*.tif (24, concatenated, sorted)` | `f614a129457f84c0c5b49c7a1a88ddfab94818c2bd96b1cb26f9e4c0155493e9` |
| `../../checkpoints/robust/p2_dropout_seed*.pt (10, concatenated, sorted)` | `8bc2646ed372b5e1c60f9a0d17f20b7735064d94d54d284bbed785e7a13f40e2` |
| `../../checkpoints/robust/p6_async_seed*.pt (10, concatenated, sorted)` | `b579af2b594ca9a0ab2b30fc660f2ac748f441fc032702717699bc27e4167abe` |
| `../../checkpoints/robust/p2_full_seed*.pt (10, concatenated, sorted)` | `5817ec427596daa2062c84902eed54e727681457e33aae4d4129b59c504b844e` |
| `../../checkpoints/robust/p6_unet_sar_seed*.pt (10, concatenated, sorted)` | `5fd758a336d0edc85eac387912a3a668e7f706384af2e7dee675213c5d75b03a` |

## 8. Deviation (2026-10-23, before any Step 8 result was saved or read)

**What happened.** The first run of `logs/robust/p8_run.sh` (22:52 IST; log kept as
`logs/robust/p8_run_run1_anchor_failed.log`) stopped at the first checkpoint: `ANCHOR FAILED p2_dropout seed 1` (tag
`oscd` and the full-input pass did not reproduce the stored test counts). The run was not saved (`phase8/runs` empty);
nothing was printed or analysed. Its test outputs for that one checkpoint were computed in memory with the defect below and
discarded unseen.

**Cause.** `eval_phase8.py` (and `eval_phase7.py`, which it imports from and which Step 7 uses) did not put the model in
evaluation mode and did not set cuDNN to deterministic, unlike training and every earlier evaluator (`eval_phase4b.py`,
`eval_phase5.py`, `eval_phase5s.py`). Diagnosed on the **validation cities only**: with `model.eval()` and
`torch.backends.cudnn.deterministic = True, benchmark = False`, `eval_phase8.run` reproduces the validation F1 that
`p2_dropout_seed1` stored at training time exactly (no_o2 0.3009471929753242, full 0.3651203112445553); without them it
does not.

**Change (approved by the author before it was made).** Those two lines were added to both evaluators; nothing else
changed (conditions, tags, thresholds, cities, analysis, audit). Hashes: `eval_phase8.py` 547e0fb0… → eb8e5295…,
`eval_phase7.py` e89089bc… → a17599a6… (rows above updated). The smoke test is unaffected (it compares no stored counts);
it is not re-run. The anchors are the check that the fix is right. Same deviation recorded in
`PHASE7_PREREGISTRATION.md` §10.
