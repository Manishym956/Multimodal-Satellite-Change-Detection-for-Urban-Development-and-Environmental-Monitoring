# Phase 3 pre-registration: post-change optical cloud corruption (evaluation only)

Written 2026-10-21, before `eval_phase3.py` was run on any checkpoint. No Phase 3 model output, validation or test,
existed when this file was written. Any later change to it must be reported as a deviation.
This file supersedes `PHASE3_PROTOCOL_DRAFT.md`; the three reviewer decisions are applied below.

## Question

Does modality-dropout training (B, `p2_dropout`) keep an advantage over full-input training (A, `p2_full`) when the
post-change optical image is **partly cloud-corrupted**, as opposed to fully missing (Phase 2)? Phase 2 already
supports B > A for the missing post image. Here the open question is the **partial** levels.

## Fixed (no training, no tuning)

- Models: the 20 Phase 2 checkpoints `checkpoints/robust/p2_{full,dropout}_seed{1..10}.pt`, unchanged and paired by seed.
- Split, SAR inputs, pre-change optical, labels, preprocessing, orbit choice (first listed) and metric code are as in Phase 2.
- **Decision 1:** only the post-change S2 image (t2) is corrupted. Corrupting both dates is a possible follow-up, not part of this phase.

## Corruption: real cloud shapes, simulated reflectance (decision 2)

Masks come from the Sentinel-2 Cloud Mask Catalogue (Francis, Mrziglod, Sidiropoulos, Muller 2020; Zenodo
10.5281/zenodo.4172871; licence CC-BY-4.0). File: `datasets/s2_cloud_mask_catalogue/masks.zip`,
sha256 `2c5c52e1…2cac74e`, 513 masks, 1022×1022 at 20 m, classes CLEAR / CLOUD / CLOUD_SHADOW. Only CLOUD is used.

**Masks** (`src/phase3/corruption.py: make_mask`):
- One mask per (city, level), for the 4 validation and 10 test cities, levels c10 / c30 / c60. Validation and test cities get separate masks.
- Seed = 20261021 + first 8 hex digits of sha256("city:level"). Each draw takes a catalogue mask uniformly at random,
  upsamples CLOUD ×2 (nearest neighbour, 20 m → OSCD 10 m), and takes a uniform window of the city's size. The draw is
  accepted when the window's cloud fraction is within ±10% (relative) of the target.
- The fraction is then made exact. The window is Gaussian-blurred (σ = 2 px), a seeded 1e-6 tie-break noise is
  added, and exactly k = round(level × H × W) pixels are marked, using numpy round (half to even).
  **"10/30/60%" therefore means exactly that fraction of each city's pixels, ±0.5 pixel.**
- The adjusted masks agree with the raw real-cloud windows on at least 94.1% of pixels (`agreement_with_raw` in the manifest).
  42 masks were drawn from 32 distinct catalogue scenes.
- Masks are saved in `experiments/robust/phase3/masks/` (sha256 of the concatenated files, sorted by name: `364ca635…b56d2125`).
  The manifest is `mask_manifest.json` (`e1c60c62…4b2b87`) and records the seed, scene, window, raw fraction and per-file hash.
- Deviation before any evaluation: the first mask build used an absolute tolerance of ±0.05. That could
  nearly double a 10% window's cloud area (chongqing 5.5% → 10%). The build was redone with the relative tolerance before
  any model saw a mask. The superseded masks are kept in `phase3/masks_superseded_tol005/` and are not used.

**Reflectance** (`corrupt_post`). It is applied after OSCD preprocessing (/10000, clip [0,1]), to the 13 S2 bands of t2 only:

  x′_b = clip((1 − α)·x_b + α·c_b, 0, 1)

- Cloud levels c10 / c30 / c60: α = 1 inside the mask (opaque) and 0 outside.
- `c100` (secondary): α = 1 everywhere.
- `haze` (secondary): α = 0.3 everywhere.
- `missing`: S2 t2 = 0. This is exactly the Phase 2 `no_o2` transformation (`apply_condition(..., "no_o2")`), and the anchor check below verifies it.
- c_b is the per-band median TOA reflectance of 8 × 50,000 CLOUD pixels from 8 catalogue scenes. The scenes were drawn with seed
  20261021 from the 96 tagged thick=1, thin=0, snow/ice=0, difficulty ≤ 2, cloud ≥ 30%, MAIN. The spectrum is clipped to [0,1]
  and recorded in `cloud_spectrum.json` (`7ee3adc4…74762`):
  B01–B12 (incl. B8A) = 0.693, 0.673, 0.619, 0.654, 0.661, 0.671, 0.681, 0.657, 0.680, 0.334, 0.039, 0.362, 0.291.
- The catalogue is L1C TOA reflectance /10000 in the same band order as OSCD (catalogue README).
- The spectrum and the α values are fixed and will not be tuned.

Known limits of this model: clouds are uniform (no texture), have hard edges, cast no shadows, and do not interact with
the surface. The masks come from other scenes. Results are a **simulated robustness test with real cloud shapes**,
not evidence about real cloudy acquisitions.

## Thresholds (decision 3)

- **Primary:** each checkpoint's stored clean-validation threshold (the Phase 2 `full` threshold), applied unchanged to every condition.
- Secondary (diagnostic): a per-condition validation argmax, chosen on the validation cities with their own corrupted masks. This
  measures how much adaptation could recover. F1 at 0.5 is also reported.
- Test cities are evaluated once per checkpoint and condition. No threshold, mask, spectrum or parameter is chosen from test data.

## Anchors (`eval_phase3.py`; a mismatch stops the run for that checkpoint)

- `clean` test counts at the stored clean threshold and at 0.5 must equal the Phase 2 `full` counts.
- `missing` test counts at the stored no_o2 threshold and at 0.5 must equal the Phase 2 `no_o2` counts.
- The clean validation argmax must equal the stored clean threshold.

## Analysis (`analyze_phase3.py`, written before evaluation)

- **Primary family:** paired B − A test F1 at the clean threshold for c10, c30, c60 and missing (n = 10 paired seeds). Paired
  t-test, with Holm correction over these 4. Reported alongside: mean Δ, SD, 95% CI, d_z, B > A count, and Wilcoxon (also Holm-adjusted).
- **Claim rule:** "B more robust at level L" only if Holm p < 0.05 and Δ > 0. The `missing` result replicates Phase 2 and is
  not new evidence. The new question is c10 / c30 / c60.
- **Secondary families**, each with its own Holm correction and labelled descriptive:
  - c100 and haze at the clean threshold;
  - all conditions at the per-condition validation threshold;
  - all conditions at 0.5.
- For each arm and condition: absolute F1 and its change from clean (paired within the arm), plus precision, recall, IoU and false-alarm rate, as mean ± sample SD.
- **Failure modes (c10–c60, clean threshold):** false-alarm rate and recall on pixels inside versus outside the mask, each compared
  with clean input on the same pixels, and the share of false positives that fall inside the mask.
- Curves: F1, precision, recall and false-alarm rate against severity, for both arms (`phase3/curves.png`).
- **Interpretation guide:** B wins at partial levels → dropout transfers to this simulated corruption. B wins only at
  missing → the advantage is specific to missing input. Both degrade alike → dropout does not address this failure.
  False positives concentrated in the mask → clouds are read as change. Recall lost mainly in the mask → change evidence is hidden.
  No outcome by itself justifies a new architecture.
- **Caution:** the 10 seeds measure training variability on the same 10 test cities. They are not independent geographic test sets.

## Code and asset hashes (sha256)

| file | sha256 |
|---|---|
| `src/phase3/corruption.py` | `6418fde31779dd219f3b2ec7204c54760b2f7b7d70bb9070db86012639fb7d1e` |
| `src/phase3/remote_zip.py` | `89292d00eace9d3d374ff2e9cd5ab32a76b23cd42da4d9faf879ea939ccf8448` |
| `build_phase3_corruptions.py` | `149328a33f9bd5bed98df8eda552cdb646435323c41f23df76b585a7a7a3ba6d` |
| `eval_phase3.py` | `5e28cf475e3f89c75601f11094bd817691c22b4f54273fd6a152f01368a673b3` |

Correction after this file was first written, before any test evaluation: a smoke test of `eval_phase3.py` on the
**validation cities only** (checkpoint `p2_full_seed1`; no test city loaded, nothing saved) found a missing `model.eval()`.
BatchNorm was in training mode, and the clean validation argmax came out as 0.99 instead of the stored 0.77. One line was added.
After the fix, validation argmaxes match the stored thresholds for `p2_full_seed1` and `p2_dropout_seed1` (clean and no_o2).
The hash above is the fixed file. The first version's hash was `cd78d297…3807cd`.
| `analyze_phase3.py` | `96bd2cc6811321ad00438f45cb81d3e01cc485d4406c2aa5fc7e770a6354a04b` |
| `src/training/train_robust.py` | `c7d812e3fd8dcf2c9bd8c7f4d3faeb1bee0c97fc256f83a76ee2194e15a6436c` |
| `phase3/mask_manifest.json` | `e1c60c62db93777a11e1d049d8266b8b5c4577449346124da7ad6f5f404b2b87` |
| `phase3/cloud_spectrum.json` | `7ee3adc47c490d2505d47231945adfc4a47c688b015dd147d1bc14f420e74762` |
| `phase3/masks/*.npy` (concatenated, sorted) | `364ca6354903f8ccad4f9bb2d2095fdfab1e2c4d79a935cec1d593edb56d2125` |
| `datasets/s2_cloud_mask_catalogue/masks.zip` | `2c5c52e1b7d7c6b64241a79b7994b732c564854afb1bac41d9aa82c832cac74e` |
