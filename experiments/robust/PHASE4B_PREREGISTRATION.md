# Step 4B pre-registration: spatial partial-dropout training (new arm C)

Written 2026-10-22 13:57 (UTC+0530), before any `p4b_spatial` training run. At writing time `checkpoints/robust/p4b_spatial_*` and
`experiments/robust/p4b_spatial_*` do not exist, and no Step 4B model output (validation or test) exists.
The mask bank (a data-independent training input, §4) was built before this file. The validation-only smoke test and the
spatial=0 parity run (§11) run after this file and before training, and their results are appended to §11 before the
first training run. Any later change must be reported as a deviation. Branch `robust-phase4`, base commit `eead015`.

## 1. Question and hypotheses

Does training with spatially localised, cloud-shaped zero-fill of the post-change optical image (arm C) make the model more
robust than whole-modality dropout (arm B) under the Step 4A / 4A-bis conditions, and what does it cost on clean input?

- **H1 (oracle fill):** C − B > 0 at z10, z30, z60. Room is small: B is 0.014 / 0.037 / 0.054 F1 below its clean F1.
- **H2 (imperfect masks, two-sided):** C − B at c30__erode4, c30__dilate4, c30__miss25. Step 4A-bis showed the residual
  failure is cloud-valued pixels that the mask misses. C never sees cloud-valued pixels in training, so it is **not
  expected** to fix the erosion/miss cost. Whether it does is the open question.
- **H3 (cost):** C − B on clean input, reported whatever its sign.

All tests are two-sided.

## 2. Arms

| arm | name | role | `--train-conditions` |
|---|---|---|---|
| B | `p2_dropout` seeds 1–10 | comparison; existing Phase 2 checkpoints, unchanged | `full=0.5,no_o2=0.25,no_opt=0.25` |
| C | `p4b_spatial` seeds 1–10 | new | `full=0.5,spatial=0.25,no_o2=0.125,no_opt=0.125` |

B is not retrained. Its stored Step 4A-bis counts (`phase4abis/runs`) are reused; they were produced by the same
`eval_phase4abis.evaluate` function that `eval_phase4b.py` uses for C. B's checkpoint hashes are in
`ARTIFACT_SHA256_2026-10-22_pre4b.txt` and in §10. A (`p2_full`) is not part of Step 4B.

**Change in exposure (limitation; C − B does not isolate spatial dropout):**

| per-sample probability during training | B | C |
|---|---|---|
| full input | 0.50 | 0.50 |
| `no_o2`: whole post-change S2 zero | 0.25 | 0.125 |
| `no_opt`: both S2 dates zero | 0.25 | 0.125 |
| `spatial`: post-change S2 zero under a cloud mask | 0 | 0.25 |
| any zero-filled post-change S2 | 0.50 | 0.50 |
| pre-change S2 zero | 0.25 | 0.125 |
| expected fraction of post-change S2 pixels zero-filled | 0.50 | 0.25 + 0.25 × 0.480 = 0.370 |

The full-input share is held fixed. C trades half of B's whole-image dropout for spatial dropout. Any C − B difference
mixes the effect of spatial masks with the effect of less whole-image dropout. The Phase 2 conditions (secondary, §7)
show what C pays on `no_o2` / `no_opt`. This design does not claim to isolate spatial dropout.

## 3. The `spatial` condition (`src/training/train_spatial.py`)

- **Per sample, independently:** each of the 64 samples in a batch draws its condition from the mixture above.
  For `spatial`, one mask is drawn uniformly, with replacement, from the bank (§4) with the run's `numpy` generator.
- **Where:** after the 32×32 importance crop and the flip / rot90 augmentation, so the mask is in crop coordinates.
  The mask is not augmented separately, because bank windows are already random positions in random scenes.
- **What:** all 13 S2 bands of the post-change image are set to 0 under the mask. S1 (both dates), S2 pre-change and the
  label are unchanged. 0 is the same fill value as Phase 2 `no_o2` and Step 4A `zero_post`, applied after OSCD
  preprocessing (/10000, clip [0, 1]). It is the fill value used here, not an intrinsic missing-data token.
- **Severity:** sampled implicitly from the bank's cloud-fraction distribution (§4). No curriculum or schedule.
- **No cloud-valued corruption is used in training.** Only zero-fill.

## 4. Mask bank (`build_spatial_mask_bank.py`, built once, refuses to overwrite)

- Source: Sentinel-2 Cloud Mask Catalogue (`datasets/s2_cloud_mask_catalogue/masks.zip`, sha256 `2c5c52e1…2cac74e`), CLOUD class only.
- **Leakage control:** every catalogue scene behind any Phase 3 mask is excluded, validation and test cities, current and
  superseded (±0.05) builds. That is 42 scenes, leaving 471 eligible. No validation or test cloud shape
  can enter training. Phase 3 / 4A / 4A-bis evaluation masks are never read by training code. The validation cities are
  used only for checkpoint and threshold selection, under clean / `no_o2` / `no_opt` inputs (no cloud masks).
- Draw (seed 20261022): uniform eligible scene, CLOUD ×2 upsampled (20 m → 10 m), uniform 32×32 window; kept if the cloud
  fraction is in [0.05, 0.95] (partial masks only). N = 20 000 kept from 446,598 draws, 153 distinct scenes;
  the largest single-scene share is 3.0%.
- **Realised severity (cloud fraction per mask):** mean 0.480; quantiles 5% 0.074, 25% 0.213, 50% 0.465, 75% 0.746, 95% 0.920.
  Histogram over [0, 0.1), …, [0.9, 1.0]: 1976 / 2749 / 2200 / 1909 / 1784 / 1789 / 1720 / 1974 / 2324 / 1575.
- Files: `phase4b/spatial_mask_bank.npy` sha256 `d1aa2b97acacd42dbbd36c87a40254e1d1f42f587b2af04279243b85559e4e29`; `phase4b/spatial_mask_bank.json` sha256 `61c945dad58b3d3e9515dad8c236a6faf5c2318f7bbfe8bf4a58c56ad06d0918`.
  Build log: `logs/robust/p4b_bank.log`.

## 5. Training (identical to B except the condition mixture)

- Network and loss: Hafner DS-UNet [64,128,256,512] and `jaccard_like_loss`, loaded from `repos/2/DS_UNet@8bfa4be`.
- Optimiser: Adam, lr 5e-5, weight decay 5e-4, no schedule. Batch 64; cities drawn uniformly with replacement from the
  10 training cities, random S1 orbit. 32×32 importance crop (20 candidates, weight = changed pixels + 5), flip + rot90.
- **Budget: 2500 optimiser steps, validation every 25 steps,** the same as B. Seeds 1–10, paired with B by seed:
  `set_seed(seed)` and `default_rng(seed)` as in B, so initial weights are identical. The data stream is identical up to the first
  `spatial` draw, and differs after it.
- **Checkpoint selection (validation cities only):** the step with the highest mean best-threshold validation F1 over
  `full`, `no_o2` and `no_opt`. This is B's rule. `spatial` is not in the selection score. There is no validation spatial
  condition, because one would need validation cloud masks, and those are evaluation masks.
- **Thresholds:** per Phase 2 condition, the validation argmax in {0, 0.01, …, 1} for the selected checkpoint. The
  **primary robustness threshold is the frozen clean (`full`) validation threshold**, as in Phase 3 / 4A / 4A-bis.
- Determinism: cudnn deterministic, no benchmark. On this machine the Phase 1 code check reproduced a run with bit-identical weights.
- Command (one per seed, sequential): `.venv/Scripts/python.exe -m src.training.train_spatial --name p4b_spatial --seed S
  --steps 2500 --eval-every 25 --train-conditions full=0.5,spatial=0.25,no_o2=0.125,no_opt=0.125` (log `logs/robust/p4b_train.log`).

## 6. Test evaluation (once per checkpoint)

1. At the end of training, `train_spatial.py` evaluates the official 10 test cities once on the Phase 2 conditions (`full`, `no_o2`,
   `no_opt`, `o2_copy`), as B's training did.
2. `eval_phase4b.py` evaluates each C checkpoint once on the 33 Step 4A-bis conditions: clean; c10/30/60 cloud with no fill;
   z10/30/60 oracle zero-fill; 27 imperfect-mask conditions. It also scores the same predictions inside / outside the true
   cloud mask M. It runs on the frozen Phase 3 masks, cloud spectrum and 4A-bis detector masks.
   Anchors (any failure stops the analysis): the clean validation argmax equals the stored threshold; clean test counts equal
   the training-time `metrics.json`; inside + outside counts equal the pooled counts.

## 7. Analysis (`analyze_phase4b.py`, fixed by hash below)

- **Primary family (Holm over 7):** paired C − B test F1 at the frozen clean threshold for
  `clean, z10, z30, z60, c30__erode4, c30__dilate4, c30__miss25`.
  These are the oracle conditions at every level plus the middle error level of each mask-error type at 30% cloud,
  the primary level of Step 4A-bis. Paired t-test (two-sided); mean Δ, 95% CI, d_z and Wilcoxon (also Holm) are reported alongside.
- **Pre-specified labels:** "C better" if Holm p < 0.05 and Δ > 0; "C worse" if Holm p < 0.05 and Δ < 0; otherwise
  "no detectable difference". **Substantial clean regression:** mean clean Δ < −0.02 F1, whatever its p-value.
- **Secondary (each its own Holm family, descriptive):** C − B on the other 26 Step 4A-bis conditions, and C − B on the
  Phase 2 conditions at each run's validation-selected threshold (from `metrics.json`).
- **Descriptive (no tests):** precision, recall, IoU and false-alarm rate for the primary conditions; inside / outside
  the true mask at 30% cloud (B from the 4A-bis exploratory addendum runs, same function and partition); selected steps.
- **Sensitivity:** with 10 pairs, power 0.8, the minimum detectable effect is 1.41 SD of the paired differences at
  α = 0.05/7 (first Holm step) and 1.00 SD at α = 0.05. Reference spreads: SD(B − A) clean 0.048; SD of B over seeds 0.028 (z30),
  0.030 (clean). So effects below about 0.04–0.07 F1 can be missed. A non-significant result is not evidence of equivalence.
- The 10 seeds are training replicates on the same 10 test cities, not independent geographic samples.

## 8. Decision guidance (pre-specified; interpretation, not a test)

- **C better on the imperfect-mask conditions, no substantial clean regression:** spatial partial-dropout training
  reduces the imperfect-mask failure. A new architecture is not yet warranted.
- **No detectable difference, or C worse, on the imperfect-mask conditions:** the residual failure is cloud-valued pixels
  reaching the model, which zero-fill training does not address. Next steps, each to be pre-registered separately:
  (i) a cloud detector biased to recall, which 4A-bis supports; (ii) training that also shows cloud-valued corruption, or an explicit
  validity-mask input channel. The validity channel is the minimal architecture change.
- **Substantial clean regression:** C is not recommended, whatever its robustness gains.
- No outcome here establishes robustness to real clouds. The corruption is simulated.

## 9. Stopping rules, failures, exclusions

- All 10 seeds are trained sequentially with the frozen code. Nothing is stopped or changed based on results. Test F1
  printed at the end of each training run is not used for any decision.
- A crashed run is restarted with the same code and seed (training is deterministic) and reported. If a seed cannot be
  completed, the analysis is not run until the issue is reported and a decision is documented.
- Any anchor failure in `eval_phase4b.py` stops the analysis.
- If the smoke test or the parity run (§11) fails, no training starts. The failure is documented, and the protocol is not changed without
  a recorded decision.

## 10. Hashes (sha256)

| file | sha256 |
|---|---|
| `src/training/train_spatial.py` | `749c98f8f7f311795ddc042a59949da168709eb8036f3c47437a29f96a06e1c7` |
| `src/training/train_robust.py` | `c7d812e3fd8dcf2c9bd8c7f4d3faeb1bee0c97fc256f83a76ee2194e15a6436c` |
| `src/datasets/oscd_dataset.py` | `bc9b64bc2fb9808efb2d8b6767a594f19385e52f4514c09633b3a7af8486fc43` |
| `src/training/train_sar_baseline.py` | `bbd0044580a207a318b452f6c153d3c67b94ea4cc1f388283f8012fecaf19ec8` |
| `src/evaluation/metrics.py` | `6d49b1892b9a2689bc8c4c23673535b06e5dc4f45df86c1bd3c0279a9f7b4854` |
| `build_spatial_mask_bank.py` | `6d69cc9c3804bef16d58afae7f7dd6404056b27d1112608084c3dc15cc63efb9` |
| `experiments/robust/phase4b/spatial_mask_bank.npy` | `d1aa2b97acacd42dbbd36c87a40254e1d1f42f587b2af04279243b85559e4e29` |
| `experiments/robust/phase4b/spatial_mask_bank.json` | `61c945dad58b3d3e9515dad8c236a6faf5c2318f7bbfe8bf4a58c56ad06d0918` |
| `smoke_phase4b.py` | `b55f8f8893619e3b5e3bf2fe7b3836ab6873ec4b8cac451975bb90c028e2bf07` |
| `eval_phase4b.py` | `9c7149861c33a514caba05027d9b239c94f2b1c8086c033a13d70bc85fe73f0d` |
| `eval_phase4abis.py` | `778de0723cd53c464137fab4c1f9aa22ec457a0679b921a9af4f3ff47a67c491` |
| `eval_phase4abis_addendum.py` | `330bf38d1d1fc5e406f0f4ac43815ed6740557e414a90d742f285f9f2fe950b1` |
| `eval_phase4a.py` | `5d831a81126b97c99778fa835ec35ef068520b30b960756c3c3b97b3ccc70f62` |
| `analyze_phase4b.py` | `4061359102730a2e3e2a4871b6515248fb57642cb9286b17cfd64ee7dd2640e6` |
| `analyze_phase3.py` | `96bd2cc6811321ad00438f45cb81d3e01cc485d4406c2aa5fc7e770a6354a04b` |
| `src/phase3/corruption.py` | `6418fde31779dd219f3b2ec7204c54760b2f7b7d70bb9070db86012639fb7d1e` |
| `src/phase3/mask_errors.py` | `7758b8cc436c83de02784a39adaf8725b15fe60b6a2437d0b2429f8807f667bd` |
| `experiments/robust/phase3/cloud_spectrum.json` | `7ee3adc47c490d2505d47231945adfc4a47c688b015dd147d1bc14f420e74762` |
| `experiments/robust/phase4abis/mask_manifest.json` | `9504e63b8fbc537374d5c6b15db9481d54e124c2c40959c9403a4bc717781709` |
| `repos/2/DS_UNet/utils/networks.py` | `a3e7216b6054fe85cb3797233b08aa08f25b4cfb1d70c2b159e00e1bb3066830` |
| `repos/2/DS_UNet/utils/loss_functions.py` | `6e71a2e830f9c39886aa114d2dc23d763e3fe653e46deae41c07f51a4f204a7f` |
| `phase3/masks/*.npy (42, concatenated, sorted)` | `364ca6354903f8ccad4f9bb2d2095fdfab1e2c4d79a935cec1d593edb56d2125` |
| `phase4abis/masks/*.npy (378, concatenated, sorted)` | `6d5b75b807147c17e9d7033c6995a539b1132d7d561bd96d24003b838b927b0f` |
| `checkpoints/robust/p2_dropout_seed1.pt (B)` | `5d4ee88e7b25dd44d005ff947e04462ad8c5a05128c18337e07c95cde4148c56` |
| `checkpoints/robust/p2_dropout_seed2.pt (B)` | `a7f653323b57b4146b21ca3667fd040b7e66d43c65747fc2ab1700d14d60f271` |
| `checkpoints/robust/p2_dropout_seed3.pt (B)` | `e6791e7c25c009223e70e80cc88a7b4ea93785a53c668b7fc44b2e6c25f9652b` |
| `checkpoints/robust/p2_dropout_seed4.pt (B)` | `26af5944cf1f589fc0307b2117ca8193e5cab083e83cbfb792ce8ab5cd09fc21` |
| `checkpoints/robust/p2_dropout_seed5.pt (B)` | `ea29c2cf5a95a1a97acd381ee8e0a4f783636f7ee7e34ae908916db3f02281e0` |
| `checkpoints/robust/p2_dropout_seed6.pt (B)` | `7f87aa281991f28da7a53577b0c0201cc4e118d4a01c178a810270b1d414e6dd` |
| `checkpoints/robust/p2_dropout_seed7.pt (B)` | `963b6ed51b3a37e440eadc17f5144860b1f60f4c01f403951366e6892b50c8a5` |
| `checkpoints/robust/p2_dropout_seed8.pt (B)` | `218bc4b88c55db3953956afd794c9578afce1471565827b01885e940c5d2fb2a` |
| `checkpoints/robust/p2_dropout_seed9.pt (B)` | `53faaf97a607fb5d39a061cdf6e84d5018d98aa4f3b9895dcf2e17954b04e093` |
| `checkpoints/robust/p2_dropout_seed10.pt (B)` | `304ecd600b7923a9da1761e5d0a4e9db81a84542d1eef93d12004c340adbaf1b` |

## 11. Pre-training verification (appended 2026-10-22 14:05 IST, before the first training run)

No setting in §1–§10 was changed after these checks.

- **Smoke test** (`smoke_phase4b.py`, sha256 in §10; log `logs/robust/p4b_smoke.log`; results `phase4b/smoke/smoke_results.json`),
  validation cities only: **21 checks, 0 failed.** It covers bank shape, fractions, hash and scene exclusion; condition-sequence identity with
  `train_robust` at spatial = 0; C-mixture frequencies; exact zero-fill under the mask in all 13 bands; S1, t1, label, input
  tensors and the dataset's cached tensors unchanged; crop/augmentation alignment; full / no_o2 / no_opt identical to
  `train_robust`; 3 GPU spatial training steps with finite loss and gradients; eval mode after `evaluate()`; checkpoint round-trip
  identity; and `eval_phase4abis.evaluate` reproducing the stored 4A-bis validation counts for `p2_dropout_seed1` (33 conditions).
- **Parity** (`logs/robust/p4b_parity.sh`, log `logs/robust/p4b_parity.log`, 13:58–14:04 IST): `train_spatial.py` with
  `full=0.5,no_o2=0.25,no_opt=0.25` (spatial = 0), seed 1, 2500 steps, reproduced `p2_dropout_seed1` **bit-for-bit**:
  weights identical, validation history identical, selected step and thresholds identical, stored test metrics identical.
  The RNG streams are therefore equivalent at spatial = 0. With spatial > 0, the extra bank-index draw shifts the stream after the
  first spatial sample, as stated in §5. The parity run's end-of-training test evaluation reproduces an existing B run and
  adds no new test information. Its outputs are `experiments/robust/parity_spatial0_seed1/` and `checkpoints/robust/parity_spatial0_seed1.pt`.
- **Analysis dry run:** `analyze_phase4b.py` was run on a temporary copy of B's stored runs labelled as C. Every Δ was 0 and every
  label was "no detectable difference". This exercised every code path without any C output. The new `eval_phase4b.py` anchor
  expressions hold on B's stored addendum counts.
- **Audit script written before training:** `audit_p4b.py` (sha256 recorded in the training commit).
