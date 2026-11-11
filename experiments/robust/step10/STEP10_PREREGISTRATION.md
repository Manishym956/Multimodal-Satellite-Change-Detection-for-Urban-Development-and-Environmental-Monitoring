# Step 10 pre-registration: confirmatory replication on fresh training seeds 11–20 (new training)

Written 2026-11-11, after Step 9 and before any Step 10 run (no `experiments/robust/p10_*` run, no
`checkpoints/robust/p10_*` checkpoint, and no run of any arm with a seed above 10 exists). Base commit `2a4cf09`
(branch `main`). This file and the scripts, smoke log and inputs in §8 are frozen by SHA-256, and the file is committed
before training starts; `run_step10.sh` refuses to start otherwise. Any later change is reported as a deviation.

## 1. Why, and what is already known

Step 9 (seeds 1–10) found, in an analysis **chosen after its results were seen** (not registered in Step 9), that E,
the early-fusion modality-dropout U-Net, evaluated with **no optical input at all** scored above the SAR-only U-Net
trained on SAR alone: E@`no_opt` − SAR-only +0.072 F1 (95% CI [+0.045, +0.099], 10/10 seeds, paired t p 0.0002), and
above B@`no_opt` (+0.050, 10/10). If it holds, training with optical data and then removing it gives a better SAR-only
detector than training on SAR alone. Because the hypothesis was selected on seeds 1–10, those seeds cannot confirm it.
Step 10 tests it on seeds that have never been trained.

**Known before writing:** every Step 6 and Step 9 result for seeds 1–10 (`experiments/robust/step9/STEP9_REPORT.md`),
including Step 9's registered H1 (E: historical image present − absent −0.091, 0/10). Nothing for any seed above 10.
The smoke test (§7) ran on relabelled seed 1–10 runs in a deleted temporary folder and printed no metric; it also
computed, but did not show, per-city counts of those already-reported seed 1–10 models.

**What a replication over seeds does and does not show.** The test cities, validation cities and training cities are
the same as in Steps 6–9. Seeds 11–20 replicate the effect over training randomness (initial weights, crops,
augmentation, condition draws), not over cities or regions. City dependence is examined separately (§4, exploratory).

## 2. Hypotheses (two-sided)

- **H1 (primary):** on seeds 11–20, does E@`no_opt` differ in test F1 from the SAR-only U-Net (full input; it sees only
  SAR)?
- **H2 (secondary):** does Step 9's registered result replicate: E@`no_o2` − E@`no_opt` (the historical image within E)?
- **H3 (secondary):** does H1 hold at the fixed threshold 0.5 (that is, not only through threshold selection)?

## 3. Arms and training (no change to any training code)

| arm | command (`src/training/train_rrd.py`) | seeds |
|---|---|---|
| E `p10_fusion_dropout` | `--arch unet_fusion --sar hafner_vv --train-conditions full=0.5,no_o2=0.25,no_opt=0.25` | 11–20 |
| SAR-only `p10_unet_sar` | `--arch unet_sar --sar hafner_vv --train-conditions full=1` | 11–20 |

These are exactly Step 9's E and Step 6's SAR-only U-Net, with only the seed changed. Recipe unchanged: `--steps 2500
--eval-every 25`, Jaccard-like loss, Adam 5e-5, weight decay 5e-4, batch 64, 32×32 importance crops, flips and
rotations, split train 10 / validation 4 / test 10 (the validation draw is fixed and does not depend on the seed).
Checkpoint: highest mean validation F1 over the trained conditions; per-condition threshold: validation-F1 argmax.
`train_rrd.py` evaluates the selected checkpoint once on the test cities at the end of each run; `run_step10.sh`
discards that console output. Arms with the same seed share the data-sampling generator, as in Step 6, so contrasts are
paired by seed. Expected cost: about 20 runs × 3.5 minutes (≈ 70 min), plus a few minutes of per-city evaluation.

## 4. Evaluation and analysis

**Per-city evaluation (`eval_step10_cities.py`).** After training, each checkpoint is loaded read-only and its (101, 4)
TP, FP, FN, TN are saved per test city and condition (E: `full`, `no_o2`, `no_opt`; SAR-only: `full`). **Anchor:** the
city counts summed over the 10 cities equal the counts `train_rrd.py` stored, at the selected threshold and at 0.5, for
every condition; on failure nothing is saved and the run stops. Checkpoint hashes are re-checked at the end.

**Analysis (`analyze_step10.py`).** Test F1 at each run's validation-selected threshold for the condition (Step 6's
rule), pooled pixel counts over the 10 test cities, paired over seeds 11–20: paired t-test, two-sided, with mean Δ,
95% CI, d_z, seeds with Δ > 0 and the Wilcoxon test alongside. Labels: "higher" / "lower" if (Holm) p < 0.05 with that
sign, else "no detectable difference". No equivalence margin is set.

- **Primary (single test, α = 0.05):** H1, E@`no_opt` − SAR-only@`full`.
- **Secondary (Holm over 2):** H2, E@`no_o2` − E@`no_opt`; H3, H1 at threshold 0.5.
- **City level (exploratory, primary contrast only, no decision):** the pooled Δ with each test city left out (range and
  jackknife SE) and the per-city Δ with unweighted sign counts, from the per-city counts at the selected thresholds.
- **Descriptive, no tests:** F1, precision, recall, IoU and false-alarm rate for both arms and every condition, at the
  selected threshold and at 0.5; selected steps and thresholds; the primary contrast over all 20 seeds (seeds 1–10
  selected the hypothesis, so this pooled figure is not confirmatory).
- **Checks inside the analysis:** shared test and validation cities; registered settings and seed for every run; every
  stored F1 equals the F1 recomputed from its stored counts.
- **Sensitivity:** with 10 pairs and power 0.8, the minimum detectable effect is 1.00 SD of the paired differences at
  α = 0.05 and 1.14 SD at α = 0.05/2.

## 5. How the result will be read (pre-specified)

- **H1 higher:** the gain of the dropout-trained early-fusion model over a SAR-only U-Net, with SAR as its only input,
  replicates over training seeds on these cities; it is reported as a confirmed result of Step 10, scoped to OSCD,
  these 10 test cities, VV, this recipe and the untuned U-Nets. The city-level analysis states whether it rests on few
  cities. The mechanism (for example regularisation or cross-modal supervision during training) is not tested here.
- **H1 not detectable:** the Step 9 observation does not replicate on new seeds; it is reported as exploratory and not
  established, and the paper does not claim it.
- **H1 lower:** reported as a failed replication with the opposite sign.
- **H2:** replication or not of Step 9's registered harm from the historical image within E; both are reported.
- Every result is reported whatever its direction; earlier registered results remain unchanged.

## 6. Stopping rules and failures

`run_step10.sh` stops if this file is not committed or has changed since its commit, if the smoke log does not end
with `n_failed 0`, if any hash below has changed, if a run does not finish, if a per-city anchor fails or if the
analysis fails. A run that does not finish is reported and re-run from scratch with the same seed (training is
deterministic). No run is excluded. `train_rrd.py` and both Step 10 scripts refuse to overwrite.

## 7. Smoke test (no new result shown)

`smoke_step10.py` → `smoke_step10.log`: **6 checks, 0 failed**. On Step 9's E and Step 6's SAR-only runs and
checkpoints for seeds 1–10, relabelled as seeds 11–20 in a temporary folder: the per-city evaluation runs and its
anchors pass for both architectures; it refuses to overwrite; the analysis runs end to end; its primary contrast
reproduces the value computed directly from Step 9's summary; its pooled city-level Δ equals the primary mean
difference; and it refuses to overwrite. The first run failed on a JSON type error in the analysis (a numpy integer
in the city sign count), fixed before the logged run; a line printing the last traceback line on failure was added to
the smoke script at the same time. The smoke test prints pass/fail only.

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
| `experiments/robust/step9/summary.json` | `fef7635a64049309408682dc65f744f2b5263c36f0c61db1ca1cc684f46c8f69` |
| `experiments/robust/step9/STEP9_REPORT.md` | `acdf0e03ed0996ff6925e12db058c6dcb0c52c172d89c3f07f32a59bfd2048f1` |
| `experiments/robust/step10/eval_step10_cities.py` | `752f2f97fd74aad45c7185edb6fab0e17130139d49ac8fd1d6d59e6abce24725` |
| `experiments/robust/step10/analyze_step10.py` | `45df028e160c4cd7b4b9ae9c1eb983b4838ea1e5f94ba5cfe8cfe5612904a817` |
| `experiments/robust/step10/smoke_step10.py` | `30e09014530a3231f9f0d1be4e8e4b431dcbd2d9c92fe139e2bfc14f11339172` |
| `experiments/robust/step10/smoke_step10.log` | `8b3871b59528f4b260c56c6bb57ca2400137487ac26570870a92807e89e12b8d` |
| `experiments/robust/step10/run_step10.sh` | `553f06779e37f833094a30ae3b5bcb654682e5b06d5989dd016a9f94b31412a2` |
| `p9_fusion_dropout_seed*/metrics.json (10, concatenated, sorted)` | `31af0e678d69c60a7e238d41d27b36879b7764f509e47dad82e6bfb9c262ca4d` |
| `p6_unet_sar_seed*/metrics.json (10, concatenated, sorted)` | `8e963cf249323b8de65f25aa023f83dc90ffc03fc9ba5617d69af8528645609a` |
