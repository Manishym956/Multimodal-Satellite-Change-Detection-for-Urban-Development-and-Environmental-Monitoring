# Step 7 pre-registration: SAR polarization, VV vs VH vs VV+VH (RRD RQ3)

Written 2026-10-23, before any Step 7 arm was trained and **before any Step 6 test result was read** (Step 6 training
was still running; its outputs are not read by this file). Base commit `5476ffe`, branch `robust-phase4`. Plan:
`docs/final/FINAL_RESEARCH_PLAN.md`. Any later change is reported as a deviation.

## 1. Question and hypotheses

- **RQ3 (primary):** with SAR as the only input, does adding VH to VV (VV+VH), or replacing VV by VH, change test F1?
  Two-sided.
- **Secondary:** the same for the dual-stream DS-UNet on full input; and whether VV+VH lowers the false-alarm rate in
  particular land-cover classes (the RRD's "discrimination between urban development and environmental pseudo-change",
  measured as where false alarms occur, not why).
- Known when writing: Song et al. (2026, Remote Sensing 18, 1103, §5.1) chose VH over VV for construction-ready bare
  land from backscatter physics, without a model ablation. Hafner et al. (2022) used VV only.

## 2. Data (downloaded and checked before this file; `download_gee.py s1`)

- Sentinel-1 VV and VH from Earth Engine with DS-UNet's own recipe (`repos/2/DS_UNet/sentinel1_download.ipynb`):
  COPERNICUS/S1_GRD, IW, same relative orbits, mean of the 3 months before OSCD date 1 and after date 2 (Mumbai's
  special window), pixels below the floor masked, unit-scaled, clamped, unmasked to 0; exported on the exact pixel grid of
  the DS-UNet VV raster for each city and orbit. VV range [−25, 5] dB as DS-UNet; **VH range [−30, 0] dB** (VH is 5–10 dB
  lower; fixed before download). A window with no scene of a polarization gives an all-zero band.
- **Checks (smoke test, below):** the exported VV equals the DS-UNet VV raster pixel for pixel in all 102 files
  (24 cities × orbits × 2 dates). VH is empty (no scene) only in `dubai_166_t1`, `lasvegas_166_t1`, `lasvegas_173_t1`,
  exactly as the metadata inventory predicted. Las Vegas is a test city whose evaluated (first) orbit has no VH at date 1;
  Dubai's evaluated orbit (130) has VH. No training or validation city lacks VH.
- Because VV is bit-identical, the existing VV models are the VV arms: `p6_unet_sar` (SAR-only U-Net, Step 6) and
  `p2_full` (DS-UNet A, Phase 2), both with `--sar hafner_vv`.
- ESA WorldCover v100 (2020) on the same grid (`download_gee.py wc`), resized to the label grid with nearest neighbour.

## 3. Arms (official DS-UNet code via `src/training/train_rrd.py`; recipe identical to Phase 2 / Step 6; full=1)

| arm | `--arch` | `--sar` | status |
|---|---|---|---|
| `p6_unet_sar` | unet_sar | hafner_vv (= VV) | Step 6 (training at the time of writing) |
| `p7_unet_vh` | unet_sar | vh | new |
| `p7_unet_vvvh` | unet_sar | vvvh | new |
| `p2_full` | dsunet | hafner_vv (= VV) | existing (Phase 2) |
| `p7_dsunet_vh` | dsunet | vh | new |
| `p7_dsunet_vvvh` | dsunet | vvvh | new |

`--steps 2500 --eval-every 25`, seeds 1–10, checkpoint and thresholds on the 4 validation cities only. The training script
evaluates the selected checkpoint on all 10 test cities once at the end of each run (pooled; includes Las Vegas with an
empty VH band); those pooled numbers are not the registered analysis.

## 4. Evaluation (`eval_phase7.py`, once per checkpoint)

Each of the 60 checkpoints (6 arms × 10 seeds) is run on each test city (first listed orbit) under every
optical-availability condition; saved: (101, 4) TP/FP/FN/TN counts per city and condition, and for full input, counts per
WorldCover class at the checkpoint's frozen full-input threshold. **Anchors** (failure: run not saved): per-city counts
summed over the 10 test cities equal the counts stored at training time, at the stored threshold and at 0.5, for every
condition; land-cover counts sum to the city's full-input counts. Prints no test metric.

## 5. Analysis (`analyze_phase7.py`, fixed by hash below)

Counts pooled over the **9 test cities with VH at both dates** (all but Las Vegas), at each checkpoint's frozen
full-input validation threshold. Paired over seeds; paired t-test, Holm within family; mean Δ, 95% CI, d_z, seeds with
Δ > 0, Wilcoxon and Holm-Wilcoxon alongside. Labels: "<first term> higher/lower" if Holm p < 0.05, else "no detectable
difference".

- **primary_rq3 (Holm over 2):** SAR-only U-Net F1: VV+VH − VV; VH − VV.
- **secondary_dsunet (Holm over 2):** DS-UNet F1: VV+VH − VV; VH − VV.
- **secondary_far (Holm over the WorldCover classes with ≥ 1% of the pooled 9-city pixels, class 0 excluded):** SAR-only
  U-Net false-alarm rate, VV+VH − VV, per class.
- **Descriptive:** F1, precision, recall, IoU, false-alarm rate per arm on the 9 cities and on all 10 (Las Vegas flagged);
  false-alarm rate and recall per class per arm; class pixel shares.
- **Sensitivity:** minimum detectable effect 1.00 SD of the paired differences at α = 0.05, 1.14 SD at α = 0.05/2.
- `audit_p7.py` (written before any Step 7 run) recomputes everything from the saved counts with numpy/scipy.

## 6. Interpretation guidance (pre-specified; not tests)

"VV+VH higher" supports RQ3 (additional polarization improves SAR change detection here); "VH higher" would agree with
Song et al.'s preference; "no detectable difference" is reported as no measurable benefit of VH on OSCD under this
protocol. Results concern OSCD urban change, 3-month mean backscatter, this recipe and these architectures.

## 7. Gates, failures and stopping

`logs/robust/p7_run.sh` starts only after Step 6 has finished (`P6 DONE`), the smoke test has 0 failures and every hash
below is unchanged. An unfinished run stops the chain and is re-run with the same seed. No run is excluded.

## 8. Smoke test (validation only)

`smoke_phase7.py` → `experiments/robust/phase7/smoke/smoke_results.json`, log `logs/robust/p7_smoke.log`: **17 checks,
0 failed** (data checks in §2; each new arm trains 20 steps and is evaluated on the validation cities; land-cover counts
sum correctly on the validation cities; the analysis and audit run end to end on synthetic counts in a temporary folder
and the audit's statistics agree with the analysis; its other failures there are the expected checks that need real
runs).

## 9. Hashes (sha256)

| file | sha256 |
|---|---|
| `src/training/train_rrd.py` | `cf918eb7481f8cd98352a4702f16913f03f373cb9d5cd077d9fe6f4b0adcbb32` |
| `src/training/train_robust.py` | `c7d812e3fd8dcf2c9bd8c7f4d3faeb1bee0c97fc256f83a76ee2194e15a6436c` |
| `src/datasets/oscd_dataset.py` | `bc9b64bc2fb9808efb2d8b6767a594f19385e52f4514c09633b3a7af8486fc43` |
| `eval_phase7.py` | `e89089bc0e761bfaa379d9498d65c56df7bd0da9b059e61435a3d6b0eb4c352f` |
| `analyze_phase7.py` | `0a9b402302fd4fab33265c1c13fdd26de4acbe104547d50352a8020bb2c555f4` |
| `analyze_phase3.py` | `96bd2cc6811321ad00438f45cb81d3e01cc485d4406c2aa5fc7e770a6354a04b` |
| `audit_p7.py` | `ce5bdb73a7a838bc119e446051a2e718f79e6875f95b29a110eb5924f1441606` |
| `smoke_phase7.py` | `7a88df27f9a15fab4e9482f1d44c74fa96498f52156b33818b812e8501c92fce` |
| `verify_prereg_hashes.py` | `7455f73c1ea05f75673583d8b939ea9af38616685c42a5eaefe372b0e85c1c4d` |
| `logs/robust/p7_run.sh` | `234beabcf95d5331642d79a1f221e5782d9d615412457ff5c1ed4235c6132209` |
| `logs/robust/p7_smoke.log` | `ce910205fe1bcd004894c437edb71a15abf4a5f145bce7f9e54b6b6d9cef3e93` |
| `phase7/smoke/smoke_results.json` | `43f1923159725491dea3186593ec41d80355ef8e4ce97b02d0134ad603bdac78` |
| `../../datasets/oscd_s1_gee/*.tif (102, concatenated, sorted)` | `ce070b3cc8f25fcff3118955e8eb81bbb111aae651008f55517058fa41e5e9a1` |
| `../../datasets/oscd_wc_gee/*.tif (24, concatenated, sorted)` | `f614a129457f84c0c5b49c7a1a88ddfab94818c2bd96b1cb26f9e4c0155493e9` |
| `../../checkpoints/robust/p2_full_seed*.pt (10, concatenated, sorted)` | `5817ec427596daa2062c84902eed54e727681457e33aae4d4129b59c504b844e` |
| `../../docs/final/FINAL_RESEARCH_PLAN.md` | `770163f6294825ce28a6325aa72f5cb2a4c5b8c61e6ebcf52db51397c9278f1c` |
