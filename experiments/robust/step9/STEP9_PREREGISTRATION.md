# Step 9 pre-registration: an early-fusion modality-dropout U-Net (RQ1 by fusion position; new training)

Written 2026-11-11, before any Step 9 arm was trained (no `experiments/robust/p9_*` run and no `checkpoints/robust/p9_*`
checkpoint exists). Base commit `260c9b2` (branch `main`, Stage 4B frozen and evaluated). This file, `analyze_step9.py`,
`smoke_step9.py`, its log and `run_step9.sh` are frozen by the SHA-256 hashes in §8, and the file is committed before
training starts. `run_step9.sh` refuses to start unless this file is committed and unchanged. Any later change is
reported as a deviation.

## 1. Why, and what is already known

In DS-UNet the optical stream never sees SAR; the two streams meet only at the final 1×1 convolution. The historical
image can therefore add only a SAR-independent term to each pixel's score, not context for interpreting the SAR change.
The Stage 3 simulated review (DA M2; Tier 3 item 16) named this as an untested alternative explanation of the RQ1 null,
and §6 of the manuscript says an early-fusion model "would separate the two explanations; we did not train one".
Step 9 trains that model.

**Known before writing (all from earlier registered or exploratory analyses; none concerns E):**

- B, historical image present − absent (`no_o2` − `no_opt`): −0.010 F1 (95% CI [−0.026, +0.007]) at each condition's own
  validation threshold (Step 6), −0.003 at 0.5, and +0.007 ([−0.004, +0.018]) at B's clean threshold (Stage 4B).
- The input-fusion U-Net trained on full input only (`p6_unet_fusion`): full-input F1 0.495 ± 0.026; its `no_o2` and
  `no_opt` scores are stored but it was never trained on missing optical input.
- Nothing about E: it has never been trained, and no validation or test score of it exists. The smoke test (§7) used an
  untrained E on validation cities only.

## 2. Questions and hypotheses (two-sided)

- **H1 (primary):** within E, does the historical image change test F1 when the post-change optical image is missing?
  Contrast E@`no_o2` − E@`no_opt`.
- **H2 (primary):** does the image's contribution differ between early fusion (E) and DS-UNet's late fusion (B)?
  Contrast (E@`no_o2` − E@`no_opt`) − (B@`no_o2` − B@`no_opt`), paired by seed.
- **H3 (secondary):** is E without the post-change image different from the SAR-only U-Net, and from B without it?
- **H4 (secondary):** on full input, does dropout training cost E anything against the full-input-only input-fusion
  U-Net, and how does E compare with A?

## 3. Arm and training (no change to any training code)

| arm | command (`src/training/train_rrd.py`) | status |
|---|---|---|
| E `p9_fusion_dropout` | `--arch unet_fusion --sar hafner_vv --train-conditions full=0.5,no_o2=0.25,no_opt=0.25` | new |
| A `p2_full`, B `p2_dropout` | Phase 2 (`train_robust.py`) | existing |
| SAR-only `p6_unet_sar`, input fusion `p6_unet_fusion` | Step 6 (`train_rrd.py`) | existing |

E is the official DS-UNet single-stream UNet (`repos/2/DS_UNet@8bfa4be`) on VV + 13 optical bands stacked at the input
(28 channels over both dates), the same network as `p6_unet_fusion`, trained with B's modality-dropout mixture. The
recipe is Step 6's unchanged: `--steps 2500 --eval-every 25`, Jaccard-like loss, Adam 5e-5, weight decay 5e-4, batch 64,
32×32 importance crops, flips and rotations, seeds 1–10, split train 10 / validation 4 / test 10. Checkpoint: highest
mean validation F1 over the trained conditions (`full`, `no_o2`, `no_opt`); per-condition threshold: validation-F1
argmax. As in Step 6, `train_rrd.py` evaluates the selected checkpoint once on the test cities at the end of each run;
`run_step9.sh` discards the console output, so no test score is shown until the analysis runs. No setting is changed
after any run starts. Expected cost: about 6 minutes per seed on the RTX 4060 laptop GPU (≈ 1 h).

**Not tested here:** the recipe was tuned on DS-UNet A, and E is not tuned separately (as for the Step 6 U-Nets).

## 4. Analysis (`analyze_step9.py`, fixed by hash below)

Test F1 at each run's validation-selected threshold for the condition (Step 6's rule), 10 test cities, pooled pixel
counts. Paired over seeds 1–10: paired t-test, two-sided, Holm within each family; mean Δ, 95% CI, d_z, seeds with
Δ > 0, Wilcoxon and Holm-Wilcoxon alongside. Labels: "higher" / "lower" if Holm p < 0.05 with that sign, else
"no detectable difference". A null is not evidence of equivalence; no equivalence margin is set.

- **primary (Holm over 2):** H1 and H2.
- **secondary_rq1 (Holm over 2):** E@`no_o2` − SAR-only@full; E@`no_o2` − B@`no_o2`.
- **secondary_full (Holm over 2):** E@full − input-fusion@full; E@full − A@full.
- **Descriptive, no tests:** H1 at threshold 0.5; F1, precision, recall, IoU and false-alarm rate for every arm and
  condition (`full`, `no_o2`, `no_opt`, `o2_copy`) at the selected threshold and at 0.5; selected steps and thresholds.
- **Checks inside the analysis:** all runs share the test and validation cities; every E run has the registered
  settings and seed; every stored F1 equals the F1 recomputed from its stored TP, FP, FN, TN.
- **Sensitivity:** with 10 pairs and power 0.8, the minimum detectable effect is 1.14 SD of the paired differences at
  α = 0.05/2 (first Holm step) and 1.00 SD at α = 0.05.

## 5. How the result will be read (pre-specified)

- **H1 higher and H2 higher:** with early fusion the historical image adds change evidence that late fusion does not
  use; the RQ1 null is then specific to DS-UNet's late fusion, and this becomes a main result of the paper.
- **H1 higher, H2 not detectable:** the image helps E; whether fusion position matters is not established.
- **H1 not detectable:** no contribution of the historical image is detected even when it can condition the SAR
  features; the RQ1 null then does not depend on late fusion alone. The 95% CI is reported as the range of contributions
  compatible with the data; the contribution is not claimed to be zero.
- **H1 lower:** the image lowers E's F1 (as it lowered the async DS-UNet's, Step 6, exploratory).
- Every result is reported whatever its direction. The registered Step 6–8 results and Stage 4B remain unchanged.
- Results concern one benchmark (OSCD, 10 test cities), VV, one early-fusion U-Net and one recipe.

## 6. Stopping rules and failures

`run_step9.sh` stops if this file is not committed or has changed since its commit, if the smoke log does not end with
`n_failed 0`, or if any hash below has changed. A run that does not finish stops the chain; it is reported and re-run
from scratch with the same seed (training is deterministic). No run is excluded. `train_rrd.py` refuses to overwrite an
existing checkpoint.

## 7. Smoke test (no new test output)

`smoke_step9.py` → `smoke_step9.log`: **6 checks, 0 failed**. (1–2) The analysis ran end to end on stand-in data
(Step 6's already-reported input-fusion runs relabelled as E, in a temporary folder), and its B within-model contrast
reproduced Step 6's audited B@`no_o2` − B@`no_opt` per seed. (3–6) An untrained E on the validation cities only: 14
channels per date, `no_opt` zeroes both optical dates and keeps SAR, `no_o2` keeps the historical image and zeroes the
post-change optical image, the historical image changes the early-fusion output, and validation evaluation returns all
four conditions.

**Line endings (found while preparing this step).** On 2026-10-25 git (`core.autocrlf=true`) rewrote some committed
files with CRLF line endings, so their on-disk hashes no longer matched earlier pre-registrations although their content
did. `src/training/train_rrd.py`, `verify_prereg_hashes.py` and `smoke_phase6.py` were restored from commit `260c9b2`
(`git show HEAD:<file>`); with CR removed each already hashed to its registered value, and `git diff` shows no change.
`smoke_phase7.py`, `smoke_phase8.py` and `docs/final/FINAL_RESEARCH_PLAN.md` have the same CRLF-only difference and
were not restored; Step 9 does not use them.

## 8. Frozen files (SHA-256)

| file | SHA-256 |
|---|---|
| `src/training/train_rrd.py` | `cf918eb7481f8cd98352a4702f16913f03f373cb9d5cd077d9fe6f4b0adcbb32` |
| `src/training/train_robust.py` | `c7d812e3fd8dcf2c9bd8c7f4d3faeb1bee0c97fc256f83a76ee2194e15a6436c` |
| `src/training/train_sar_baseline.py` | `bbd0044580a207a318b452f6c153d3c67b94ea4cc1f388283f8012fecaf19ec8` |
| `src/datasets/oscd_dataset.py` | `bc9b64bc2fb9808efb2d8b6767a594f19385e52f4514c09633b3a7af8486fc43` |
| `src/evaluation/metrics.py` | `6d49b1892b9a2689bc8c4c23673535b06e5dc4f45df86c1bd3c0279a9f7b4854` |
| `repos/2/DS_UNet/utils/networks.py` | `a3e7216b6054fe85cb3797233b08aa08f25b4cfb1d70c2b159e00e1bb3066830` |
| `repos/2/DS_UNet/utils/loss_functions.py` | `6e71a2e830f9c39886aa114d2dc23d763e3fe653e46deae41c07f51a4f204a7f` |
| `analyze_phase3.py` | `96bd2cc6811321ad00438f45cb81d3e01cc485d4406c2aa5fc7e770a6354a04b` |
| `verify_prereg_hashes.py` | `7455f73c1ea05f75673583d8b939ea9af38616685c42a5eaefe372b0e85c1c4d` |
| `experiments/robust/step9/analyze_step9.py` | `69dcf48302271bccd6c502702ff4580afcb3f140e71343383abca65905cc2b33` |
| `experiments/robust/step9/smoke_step9.py` | `256c704f3325e4a3685d27088af4f6aa5e2b979d0146b012bd1d5edf9165431a` |
| `experiments/robust/step9/smoke_step9.log` | `b2517cfe0dc0ef368cb1a71ffe8b67488661e3e53bf7e7e4a2743b06bb9bdc9b` |
| `experiments/robust/step9/run_step9.sh` | `21c338bafc1b680dd4d3b1bb055f030c57565bee8c8b46644705a6a9ae58b315` |
| `p2_full_seed*/metrics.json (10, concatenated, sorted)` | `856a67d41a7e20d9f6690551a9f6322eab38f9a8dc823d56390e30a110963dd1` |
| `p2_dropout_seed*/metrics.json (10, concatenated, sorted)` | `e2af8d130d433b912df9710fedc68927c4edb4a9a9a718b445ac1ff4d31e6f70` |
| `p6_unet_sar_seed*/metrics.json (10, concatenated, sorted)` | `8e963cf249323b8de65f25aa023f83dc90ffc03fc9ba5617d69af8528645609a` |
| `p6_unet_fusion_seed*/metrics.json (10, concatenated, sorted)` | `3c5d5475bb2f08c155a37edb5887a193e85334cbd6b40ea3091934ad827b5148` |
