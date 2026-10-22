# Step 4A-bis pre-registration: zero-fill under imperfect cloud masks (evaluation only)

Written 2026-10-21, after a validation-only smoke test and before any Step 4A-bis test evaluation (`phase4abis/runs/` did not exist).
Any later change must be reported as a deviation.

## Question

Step 4A used an oracle mask. How quickly does the zero-fill benefit decay when the cloud mask is wrong in controlled ways:
cloud edges missed (erosion), clear pixels masked (dilation), or whole clouds missed?

## Fixed

- The 20 Phase 2 checkpoints, unchanged. No training or tuning.
- The Phase 3 oracle masks M, cloud spectrum and corruption code are unchanged. The image is always corrupted with the **true** M.
  Zero-fill uses a **detector mask** P. Pixels in M \ P keep their cloud values; pixels in P \ M are clear pixels set to zero.
- Detector masks (`src/phase3/mask_errors.py`, built once by `build_phase4abis_masks.py`, 378 files):
  - `erode2/4/8`: M eroded by a disk of radius 2, 4 or 8 px (20, 40 or 80 m).
  - `dilate2/4/8`: M dilated by the same disks.
  - `miss10/25/50`: exactly round(Q·|M|) cloud pixels unmasked. Whole clouds are dropped in a seeded order, skipping any that would
    overshoot; any remaining deficit is taken as one contiguous region of the largest remaining cloud.
    Seed = 20261021 + first 8 hex digits of sha256("city:level:variant").
- Mean test-city error magnitudes are recorded in the manifest. For example, at 30% cloud erode2/4/8 miss 11/21/40% of cloud pixels, and
  dilate2/4/8 mask 5/10/20% of clear pixels.
- Levels: c10, c30, c60.
- Reference conditions, re-evaluated and anchored to the stored Step 4A counts on validation and test: `clean`, `cL` (cloud, no fill),
  and `zL` (oracle fill).
- Thresholds: **primary** is the frozen clean-validation threshold. 0.5 and the per-condition validation argmax are reported as secondary.
- Test cities are evaluated once per checkpoint and condition.

## Analysis (`analyze_phase4abis.py`, written before evaluation)

- **Primary family:** dropout arm B at 30% cloud. Paired F1(c30 with imperfect mask v) − F1(z30, oracle mask), same checkpoint,
  for the 9 variants. Paired t-test with Holm over 9. Mean Δ, 95% CI, d_z and Wilcoxon (also Holm-adjusted) are reported alongside.
  This measures how much each kind of mask error **costs** relative to the oracle.
- **Descriptive label, defined in advance:** retained gain = (F1_v − F1_cloud) / (F1_oracle − F1_cloud) per seed.
  "Most of the gain retained" means a mean ≥ 0.75. It is a label, not a test.
- **Secondary,** each its own Holm family and descriptive:
  - B at c10 and c60;
  - A at every level;
  - B − A under each imperfect mask at c30;
  - false-alarm rate and recall on missed-cloud pixels (M \ P) and on over-masked clear pixels (P \ M), clean input versus imperfect-mask input;
  - all conditions at the other two thresholds.
- **Caution:** the mask errors are synthetic and do not model a specific cloud detector. Real detector errors also include
  shadows and thin cirrus, which are not simulated. The 10 seeds are training replicates on the same 10 test cities.

## Hashes (sha256)

| file | sha256 |
|---|---|
| `src/phase3/mask_errors.py` | `7758b8cc436c83de02784a39adaf8725b15fe60b6a2437d0b2429f8807f667bd` |
| `build_phase4abis_masks.py` | `113fc5708dedb1df9b7e692301a6ebe50430af43da28091171b57b4bf538dc90` |
| `eval_phase4abis.py` | `778de0723cd53c464137fab4c1f9aa22ec457a0679b921a9af4f3ff47a67c491` |
| `analyze_phase4abis.py` | `014d028166c70ea04f1741a44fddc3507ce7ba167e6ef3b1cd841fd833a0df4e` |
| `eval_phase4a.py` (zero_post imported) | `5d831a81126b97c99778fa835ec35ef068520b30b960756c3c3b97b3ccc70f62` |
| `phase4abis/mask_manifest.json` | `9504e63b8fbc537374d5c6b15db9481d54e124c2c40959c9403a4bc717781709` |
| `phase4abis/masks/*.npy` (concatenated, sorted) | `6d5b75b807147c17e9d7033c6995a539b1132d7d561bd96d24003b838b927b0f` |
| Phase 3 oracle masks (unchanged) | `364ca6354903f8ccad4f9bb2d2095fdfab1e2c4d79a935cec1d593edb56d2125` |
