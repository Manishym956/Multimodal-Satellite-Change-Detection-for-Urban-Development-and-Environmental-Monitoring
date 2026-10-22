# Step 5S pre-registration: recall-biased dilation of under-masking detector masks (synthetic, evaluation only)

Written 2026-10-22 17:13 IST, before `eval_phase5s.py` ran on any test city. At writing time `experiments/robust/phase5s/runs/`
does not exist. The only model outputs produced so far are a **validation-only** smoke test (§7). Runs in parallel with, and
independent of, the real-detector Step 5 (`PHASE5_PREREGISTRATION.md`, in preparation). Any later change must be reported as a
deviation. Branch `robust-phase4`, base commit `b07bd4a`.

## 1. Question and hypothesis

Step 4A-bis showed, for the dropout-trained model B, that under-masking (eroded cloud edges, missed whole clouds) is expensive
and over-masking is nearly free. A practical remedy is to bias a detector toward recall by dilating its mask. **H (primary):**
dilating each under-masking mask P by a validation-selected radius r* changes B's test F1 at 30% cloud, relative to P itself.
Two-sided. Expected from 4A-bis: a gain for eroded masks (dilation restores edges); little or none for missed whole clouds
(dilation cannot reach clouds far from any detection). This is a synthetic test; it does not involve a real detector.

## 2. Models and inputs (no training, no new data)

- The 30 frozen checkpoints `checkpoints/robust/{p2_full, p2_dropout, p4b_spatial}_seed{1..10}.pt` (hashes in
  `ARTIFACT_SHA256_2026-10-22_post4b.txt`). **B (p2_dropout) is primary**; A and C are secondary.
- Corruption: the Phase 3 corruption (true mask M from `phase3/masks`, the Phase 3 cloud spectrum), levels c10 / c30 / c60.
- Under-masking masks P: the frozen Step 4A-bis masks `phase4abis/masks/{city}_{level}_{v}.npy`, v ∈ {erode2, erode4,
  erode8, miss10, miss25, miss50}.
- Recall-biased masks: P dilated by a disk of radius r (10 m pixels), r ∈ {0, 1, 2, 3, 4, 6, 8, 12, 16}; r = 0 is P.
- Threshold: each checkpoint's frozen clean-validation threshold.

## 3. Evaluation (`eval_phase5s.py`, once per checkpoint)

Conditions: clean, cL (no fill), zL (oracle fill), and L__v__r for every level, variant and radius (169 conditions), with
zero-fill under the mask (`eval_phase4a.zero_post`). Pooled counts on the 4 validation and 10 test cities.
Anchors (failure: run not saved, analysis stops): clean, cL, zL and every L__v__0 equal the stored Step 4A-bis (A, B) or
Step 4B (C) counts exactly, on validation and test; the clean validation argmax equals the stored threshold.
The script prints no test metric.

## 4. Radius selection (validation only)

r* = the radius with the highest mean (over B's 10 seeds) of B's mean validation F1 across all 18 under-masking conditions
(3 levels × 6 variants), at the frozen clean threshold; ties: smaller r. One r* for all arms. No test output is used.

## 5. Analysis (`analyze_phase5s.py`, fixed by hash below)

- **Primary (B, test, c30, Holm over 6):** paired F1(c30__v__r*) − F1(c30__v__0) for the 6 variants. Paired t-test,
  two-sided; mean Δ, 95% CI, d_z, seeds with Δ > 0 and Holm-corrected Wilcoxon reported alongside.
- **Labels:** "dilation better" (Holm p < 0.05, Δ > 0); "dilation worse" (Holm p < 0.05, Δ < 0); otherwise "no detectable difference".
- **Secondary (each its own Holm family, descriptive):** the same contrast for B at c10 and c60 (12); for A (18) and for C (18);
  for B, F1(L__v__r*) − F1(zL), the remaining cost relative to the oracle mask (18).
- **Descriptive:** test F1 for every radius (the full curve), retained gain (F1 − F1_cloud) / (F1_oracle − F1_cloud) at r = 0 and r*.
- **Sensitivity:** with 10 pairs, power 0.8, the minimum detectable effect is about 1.00 SD of the paired differences at α = 0.05 and
  about 1.37 SD at α = 0.05/6 (first Holm step; noncentral t). A null is not equivalence.
- `audit_p5s.py` (written before the evaluation) recomputes every number from the raw counts.

## 6. Limitations (stated before the results)

- **Synthetic on both sides:** the clouds are the flat Phase 3 spectrum, and the detector errors are morphological (erosion) or
  whole-cloud removal, not a real detector. The real-detector question is Step 5.
- **r* is shared across variants and levels,** chosen on 4 validation cities whose masks were built the same way as the test masks.
- The 10 seeds are training replicates on the same 10 test cities, not independent geographic samples.
- Zero is the fill value used here, not an intrinsic missing-data token.

## 7. Pre-evaluation verification

Validation-only smoke test (`logs/robust/p5s_smoke.log`, `phase5s/smoke_results.json`), p2_dropout_seed1 and p4b_spatial_seed1:
validation counts for every reference and every r = 0 condition equal the stored counts; clean argmax equals the stored threshold;
evaluation is deterministic; model in eval mode; dilation is monotone in r; r = 0 masks equal the 4A-bis masks; **0 failed**.

## 8. Hashes (sha256)

| file | sha256 |
|---|---|
| `eval_phase5s.py` | `35326afa26a61ff8249a5e4a4753b4100e9175d65bdbbec775a782d475141bbb` |
| `analyze_phase5s.py` | `d3d25645a2b0255eb14a456ad6b3d71b5d6432a127131fde0edce0e2761e859d` |
| `audit_p5s.py` | `f2ed56daa1037890ec7c2948df3de63101558ffdbb3deba52f0ff66e1cdc777e` |
| `eval_phase4a.py` | `5d831a81126b97c99778fa835ec35ef068520b30b960756c3c3b97b3ccc70f62` |
| `analyze_phase3.py` | `96bd2cc6811321ad00438f45cb81d3e01cc485d4406c2aa5fc7e770a6354a04b` |
| `src/phase3/corruption.py` | `6418fde31779dd219f3b2ec7204c54760b2f7b7d70bb9070db86012639fb7d1e` |
| `src/phase3/mask_errors.py` | `7758b8cc436c83de02784a39adaf8725b15fe60b6a2437d0b2429f8807f667bd` |
| `src/training/train_robust.py` | `c7d812e3fd8dcf2c9bd8c7f4d3faeb1bee0c97fc256f83a76ee2194e15a6436c` |
| `src/datasets/oscd_dataset.py` | `bc9b64bc2fb9808efb2d8b6767a594f19385e52f4514c09633b3a7af8486fc43` |
| `experiments/robust/phase3/cloud_spectrum.json` | `7ee3adc47c490d2505d47231945adfc4a47c688b015dd147d1bc14f420e74762` |
| `experiments/robust/phase5s/smoke_results.json` | `191f43c685873b0d731eb7650d5baf01821d38811591f3ae29d169d54555248f` |
| `logs/robust/p5s_smoke.log` | `90af1e826961e9c7c28094a2be21d65007c82ce67b967d68e108dc94350f40b4` |
| `phase3/masks/*.npy (42, concatenated, sorted)` | `364ca6354903f8ccad4f9bb2d2095fdfab1e2c4d79a935cec1d593edb56d2125` |
| `phase4abis/masks/*.npy (378, concatenated, sorted)` | `6d5b75b807147c17e9d7033c6995a539b1132d7d561bd96d24003b838b927b0f` |
