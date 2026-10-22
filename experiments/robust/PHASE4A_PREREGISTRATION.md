# Step 4A pre-registration: oracle-mask zero-fill (evaluation only)

Written 2026-10-21, after a validation-only smoke test and before any Step 4A test evaluation (`experiments/robust/phase4a/`
did not exist). Any later change must be reported as a deviation.

## Question

The models read cloud-valued pixels as change (Phase 3). If the same pixels are marked as missing instead (zero-filled
under a known cloud mask), does performance recover? For which training strategy?

## Fixed

- Checkpoints: the 20 Phase 2 checkpoints, unchanged. No training, no tuning.
- Masks: the Phase 3 masks (`phase3/masks/`, sha256 of the concatenated files `364ca635…b56d2125`), unchanged, so 10/30/60% means the exact fraction of each city's pixels.
- Cloud spectrum: unchanged (`phase3/cloud_spectrum.json`).
- Split, SAR, pre-change optical, preprocessing and orbit are as in Phases 2–3.
- Post-change optical inputs:
  - `clean`;
  - `c10/c30/c60`: the Phase 3 cloud corruption, same code;
  - `z10/z30/z60`: S2 t2 set to 0 **only inside the same mask** and left unchanged outside. This is the Phase 2 `no_o2` fill, restricted to the mask.
- The mask is an **oracle**: it is the exact set of corrupted pixels, so these results are an upper bound for any real cloud detector.
- Thresholds:
  - **Primary:** each checkpoint's frozen clean-validation threshold (the Phase 2 `full` threshold).
  - **Secondary:** the per-condition validation argmax, and 0.5.
- Anchors (enforced in code): `clean` and `c*` test counts must equal the stored Phase 3 counts (pooled, inside and outside the mask), and the clean validation argmax must equal the stored threshold.
- The validation-only smoke test reproduced the Phase 3 validation counts for `clean` and `c*`, and the stored clean threshold, on `p2_full_seed1` and `p2_dropout_seed1`.

## Analysis (`analyze_phase4a.py`, written before evaluation)

- **Primary family (6 tests, Holm):** the within-arm paired difference F1(zL) − F1(cL), same checkpoint, at the frozen clean
  threshold, for A and B and L ∈ {10, 30, 60}. Paired t-test. Mean Δ, 95% CI, d_z, count of seeds with z > c, and Wilcoxon (also Holm-adjusted) are reported alongside.
- **Claim rule:** "masked zero-fill improves on cloud for arm X at level L" only if Holm p < 0.05 and Δ > 0.
- **Descriptive label, defined in advance:** recovery = (F1_z − F1_c) / (F1_clean − F1_c) per seed. "Substantial" means a mean recovery ≥ 0.5.
  It is a label, not a test.
- **Secondary:**
  - B − A under z10/z30/z60 at the clean threshold (Holm over 3), and the same comparison at the per-condition validation threshold and at 0.5 (each its own Holm family);
  - false-alarm rate and recall inside versus outside the mask (clean → cloud → zero-fill on the same pixels);
  - absolute F1, precision, recall, IoU and false-alarm rate for every condition and threshold.
- **Interpretation guide:**
  - Zero-fill restores performance for B but not A → cloud-aware missing-data handling plus dropout training is the cheap remedy to pursue (Step 4B).
  - Zero-fill does not restore performance → the failure involves more than cloud reflectance, for example partial-image inputs never seen in training.
  - No outcome by itself justifies a new architecture.
- **Caution:** the 10 seeds are training replicates on the same 10 test cities. The mask is an oracle.

## Hashes (sha256)

| file | sha256 |
|---|---|
| `eval_phase4a.py` | `5d831a81126b97c99778fa835ec35ef068520b30b960756c3c3b97b3ccc70f62` |
| `analyze_phase4a.py` | `528ffa72a82ce274adf07a3311d3a605c59c303cce8aa29f7f17ea6c68a0e0ba` |
| `analyze_phase3.py` (helpers imported) | `96bd2cc6811321ad00438f45cb81d3e01cc485d4406c2aa5fc7e770a6354a04b` |
| `src/phase3/corruption.py` | `6418fde31779dd219f3b2ec7204c54760b2f7b7d70bb9070db86012639fb7d1e` |
| `src/training/train_robust.py` | `c7d812e3fd8dcf2c9bd8c7f4d3faeb1bee0c97fc256f83a76ee2194e15a6436c` |
