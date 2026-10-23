# Step 5 pre-registration: recall-biased real-detector cloud masks on composite clouds (evaluation only)

Written 2026-10-23 11:38 (UTC+0530), before any Step 5 model evaluation, validation or test. At writing time
`experiments/robust/phase5/runs/`, `operating_points.json` and `detector_outputs/test/` do not exist, and no change model
has been run on any composite. Built before this file and hashed in §11 (data-independent of any model output):
the composite inputs, the detector inputs for both splits, and the detector outputs on the **validation** split only.
Any later change must be reported as a deviation. Branch `robust-phase4`, base commit `3c5d85a`.

## 1. Question and hypotheses

Steps 4A and 4A-bis showed that for the dropout-trained model B, zero-filling cloud pixels recovers most of the F1 lost to
simulated opaque clouds, that over-masking is almost free (≤ 0.010 F1 at 30% cloud for 2–8 px dilation) and that
under-masking is expensive (0.090–0.250). That was under synthetic mask errors and a single flat cloud spectrum. Step 5 asks
whether the conclusion holds with **real cloud pixels and a real cloud detector**:

- **H1 (primary):** with a real detector (s2cloudless), a **recall-biased** operating point (rb95) gives higher F1 for B than
  the operating point that maximises mask accuracy (bal). Two-sided.
- **H2:** how much of the oracle-mask F1 does rb95 retain (rb95 − oracle), and does it beat no fill (rb95 − cloud)? Two-sided.
- **H3 (secondary):** the same contrasts for A (full-only) and C (spatial dropout; flagged, §3); the cost of false detections on
  clean images; other operating points.

All tests are two-sided.

**Known when this was written:** the Step 5S results (synthetic dilation of under-masking masks; commits `cde5434`, `3c5d85a`).
Dilation recovered most of the erosion loss but almost none of the loss from missed clouds, and Step 5S's validation-selected
radius (16 px) was at the edge of its grid. The Step 5 design (grid, rules, conditions, contrasts) was drafted before those
results and **was not changed in response to them**. In particular, the radius grid still ends at 16 px. If a Step 5 operating
point selects r = 16, that is reported as a grid-boundary limitation.

## 2. Models (no training)

The 30 frozen checkpoints `checkpoints/robust/{p2_full, p2_dropout, p4b_spatial}_seed{1..10}.pt`, unchanged (sha256 in
`ARTIFACT_SHA256_2026-10-22_post4b.txt`). **B (p2_dropout) is the primary arm.** A and C are secondary. Split, SAR inputs,
pre-change optical, labels, preprocessing, orbit (first listed) and metric code are as in Steps 3–4B. Each checkpoint's frozen
clean-validation threshold is the primary threshold.

## 3. Composite clouds (`src/phase5/composite.py`, `build_phase5_composites.py`)

Real cloud pixels from the Sentinel-2 Cloud Mask Catalogue (Francis et al. 2020, Zenodo 4172871, CC-BY-4.0) are pasted into
the OSCD post-change image:

- For each validation and test city, level L ∈ {10, 30, 60}% and replicate k ∈ {0, 1, 2} (**126 composites**: 36 validation,
  90 test), a catalogue scene is drawn uniformly from the split's pool with seed `20261023 + sha256("p5:city:level:k")[:8]`,
  and a uniform 20 m window covering the city after ×2 nearest-neighbour upsampling to 10 m. The draw is accepted if the
  window's CLOUD fraction is within ±10% (relative) of L. The window is not smoothed or adjusted, so pixel values and labels stay paired;
  the realised fractions are recorded (§11).
- The true mask M is the upsampled CLOUD window. Inside M, the post-change optical image is replaced by the catalogue
  subscene's 13 TOA bands (same band order as OSCD; ×2 upsampled; clipped to [0, 1] like OSCD preprocessing). Outside M it is
  unchanged. SAR and the pre-change image are never changed. CLOUD_SHADOW is not transplanted.
- **Scene pools.** Only difficulty-5 scenes ("possibly large errors") are excluded. The rest is split into a validation and a
  test pool by the parity of sha256("p5:" + scene). **Validation and test composites never share a scene**, so the detector's
  operating point is not tuned on test clouds.
- **Design decision recorded before any build (2026-10-22, by the author):** the first design also excluded the scenes behind
  Phase 3 masks, the Step 4B mask bank and the Phase 3 cloud spectrum. That left **no** partly cloudy scene: the catalogue has
  142 scenes with 5–95% cloud, and the bank (102) and Phase 3 (40) used all of them. The composite build failed (no acceptable
  window) before any composite was written. Excluding Phase 3 scenes was never needed for leakage (they were used only for
  evaluation masks), and A and B never trained on catalogue data. **Only arm C trained on catalogue data**: binary 32×32 cloud
  shapes from the 153 bank scenes (list recovered by an exact replay of the bank builder, §11). So these scenes are kept and
  flagged per composite (`in_bank`, `in_phase3`, `in_spectrum`), C's results are secondary and flagged, and C − B is also
  reported, descriptively, on the test composites whose scene is not in the bank.
- Subscenes are read from Zenodo by HTTP range (`src/phase3/remote_zip.py`; zipfile checks each member's CRC-32, recorded).

## 4. Detector (`export_phase5_detector_inputs.py`, `detect_phase5.py`)

- s2cloudless 1.7.3 in a separate venv (`.venv_cloud`, package list in `phase5/detector_env_freeze.txt`), so the frozen
  evaluation environment is unchanged. The packaged LightGBM model is loaded from its text (the C loader cannot open this
  project's non-ASCII path); its sha256 is recorded.
- Input: the 10 s2cloudless bands of the post-change image (composite or clean) on the 10 m grid. Output: the raw cloud
  probability map, and the library's default mask (threshold 0.4, average_over 1, dilation_size 1).
- **Operating point (t, r):** P = (prob ≥ t) dilated by a disk of radius r (10 m pixels), over t ∈ {0.05, 0.10, …, 0.90} and
  r ∈ {0, 1, 2, 3, 4, 6, 8, 12, 16}.

## 5. Operating-point selection (`select_phase5_operating_points.py`; validation composites only; no change model)

Pixel counts of P against M pooled over all 36 validation composites:

- **rb95 (primary), rb90, rb99:** among grid points with pooled cloud recall ≥ 0.95 / 0.90 / 0.99, the highest pooled precision;
  ties: larger t, then smaller r. If no point reaches the target, the highest-recall point, flagged and reported.
- **bal:** the highest pooled mask F1; ties as above.

Rationale for 0.95: in Step 4A-bis, B lost about 0.09–0.10 F1 when 10–12% of cloud pixels were missed; 95% recall halves the
missed fraction. The choice is fixed here and not revisited. No change-model output is used to choose any operating point.

## 6. Test evaluation (`eval_phase5.py`; once per checkpoint)

Conditions per level L (counts pooled over the 10 test cities × 3 replicates): `L__cloud` (composite, no fill),
`L__oracle` (zero-fill under M), `L__rb95`, `L__rb90`, `L__rb99`, `L__bal`, `L__default` (zero-fill under the detector mask).
Per city on the clean image: `clean`, and `clean__{rb95, rb90, rb99, bal, default}` (zero-fill under the detector mask run on the
clean image). Validation is evaluated the same way (anchor and threshold diagnostics only).
Anchors (failure: run not saved, analysis stops): clean validation and test counts equal the stored Step 4A-bis (A, B) or
Step 4B (C) counts; clean validation argmax equals the stored threshold; inside + outside M equals pooled; outside M each
composite equals the clean image. The script prints no test metric.

## 7. Analysis (`analyze_phase5.py`, fixed by hash below)

- **Primary family (B, Holm over 9):** paired test F1 at the frozen clean threshold, for L ∈ {c10, c30, c60}:
  rb95 − bal, rb95 − oracle, rb95 − cloud. Paired t-test (two-sided); mean Δ, 95% CI, d_z, count of seeds with Δ > 0 and
  Wilcoxon (also Holm) reported alongside.
- **Labels:** "rb95 better" if Holm p < 0.05 and Δ > 0; "rb95 worse" if Holm p < 0.05 and Δ < 0; otherwise "no detectable difference".
- **Secondary (each its own Holm family, descriptive):** the same 9 contrasts for A and for C; for B, rb90 / rb99 / default
  minus bal at each level; C − B at clean and at L__rb95; for B, clean__d − clean for each detector mask d.
- **Descriptive (no tests):** F1, precision, recall, IoU, false-alarm rate for every condition and arm; retained gain
  (F1_mask − F1_cloud) / (F1_oracle − F1_cloud) per seed; F1 / recall / false-alarm rate inside and outside M; the detector masks'
  recall, precision and over-masked fraction on the test composites (no model); C − B pooled over the test composites whose
  scene is not in the Step 4B bank.
- **Sensitivity:** with 10 pairs, power 0.8, the minimum detectable effect is about 1.00 SD of the paired differences at
  α = 0.05 and about 1.46 SD at α = 0.05/9 (first Holm step). A non-significant result is not evidence of equivalence.
- The 10 seeds are training replicates on the same 10 test cities, not independent geographic samples.

## 8. Decision guidance (pre-specified; interpretation, not a test)

- **rb95 better than bal at 30% and rb95 retains most of the oracle gain (mean retained gain ≥ 0.75):** a recall-biased real
  detector with zero-fill is a working remedy for B under these composites. A new architecture is still not warranted; the
  next step is a test on real cloudy acquisitions.
- **No detectable difference between rb95 and bal:** with this detector, the recall bias that the synthetic study favoured
  makes no measurable difference; report whether both retain most of the oracle gain.
- **rb95 worse than bal, or rb95 retains little of the oracle gain (< 0.5) at 30%:** real detector errors (e.g. thin cloud and
  edges the detector cannot reach at any threshold, or false detections) limit the zero-fill route. Next, to be pre-registered
  separately: a validity-mask input channel or training with cloud-valued corruption.
- No outcome establishes robustness to real cloudy acquisitions (§9).

## 9. Limitations (stated before the results)

- **Composites, not real acquisitions.** Cloud pixels are real, but they are pasted from other scenes: thin cloud carries the
  foreign surface, there are no matching shadows, the cloud edge is a hard boundary between two scenes, and the atmosphere outside
  the cloud is unchanged. Only the post-change date is clouded.
- **The detector is run on the composite,** so its errors are real errors on real cloud pixels, but its false detections on clear
  pixels are on OSCD surfaces, and on the clean images it may flag real (unlabelled) clouds.
- **Labels are the catalogue's semi-automatic masks** at 20 m, upsampled; boundary errors in M count as detector errors.
- **One detector** (s2cloudless, run at 10 m on 10 bands), one zero-fill strategy, one architecture.
- **Arm C may be advantaged** on composites from bank scenes, whose cloud shapes it saw (as 32×32 binary masks) in training.
- Zero is the fill value used here, not an intrinsic missing-data token.

## 10. Stopping rules, failures, exclusions

- Every checkpoint is evaluated once with the frozen code. Nothing is stopped or changed based on results.
- Any anchor failure stops the analysis until it is reported and a decision is documented.
- If operating-point selection cannot reach the 0.95 recall target, the flagged fallback (§5) is used and reported; the protocol is
  not otherwise changed.
- If the smoke test (§12) fails, no test evaluation starts; the failure is documented.

## 11. Hashes and pre-registration-time facts

**Scene pools.** The catalogue has 513 scenes; 22 difficulty-5 scenes are excluded; the remaining 491 are split by hash parity
into a validation pool of 238 and a test pool of 253 (disjoint). The 126 composites use 27 distinct validation scenes and 45
distinct test scenes; no scene appears in both splits (checked by the builder and independently).

**Composites (`composite_manifest.json`).** 126 = 36 validation (4 cities × 3 levels × 3 replicates) + 90 test (10 × 3 × 3).
Every realised cloud fraction is within ±10% (relative) of its level:

| split | c10 (min–max, mean) | c30 | c60 |
|---|---|---|---|
| validation (12 each) | 0.092–0.110, 0.100 | 0.272–0.326, 0.307 | 0.540–0.620, 0.573 |
| test (30 each) | 0.090–0.110, 0.100 | 0.271–0.328, 0.299 | 0.544–0.655, 0.594 |

Acceptance needed up to 935 draws per composite. Subscenes were read by HTTP range from Zenodo (72 distinct subscenes; each
member's CRC-32 checked by zipfile and recorded). `at_upper_clip_fraction` (share of stored band values equal to 1, the clip
bound, over all stored 20 m band values, including outside the cloud) has mean 0.005 and maximum 0.224 (norcia_c10_k1).

**Flags.** Composites whose scene is in the Step 4B mask bank: 20 of 36 validation, 56 of 90 test (so **34 test composites are
not from bank scenes**; the descriptive C − B contrast of §7 uses these). Scene behind a Phase 3 mask: 16 validation, 34 test.
Scene in the Phase 3 cloud spectrum: 0. The bank scene list (153 scenes) was recovered by an exact replay of the Step 4B bank
builder (`replay_spatial_mask_bank_sources.py`, log `logs/robust/p5_bank_replay.log`): 446,598 draws, replayed bank sha256
`d1aa2b97…` equal to the recorded bank.

**Build interruptions (no composite was regenerated or altered).**
1. Run 1 (2026-10-22, log `logs/robust/p5_composites_run1_hung.log`) wrote 87 composites, then hung from 19:26 to 20:38 IST on a
   Zenodo range read (`src/phase3/remote_zip.py` has no timeout; process at 0% CPU) and was stopped. The builder was then
   changed, before any further composite was written: 120 s socket timeout; up to 5 read attempts with the archive reopened;
   resume mode (an existing file is kept only if its cloud mask equals the redrawn window and its shapes, dtype and value range
   are right; listed in `resumed_files` and per entry); and `clipped_fraction` replaced by `at_upper_clip_fraction`, computed
   from the stored bands, for every entry. `remote_zip.py` (committed Phase 3 code) was not changed.
2. Run 2 (resume) wrote composites 88–89 and ended with the working session at 20:47 IST (no traceback, no failed read; log kept
   as `logs/robust/p5_composites_run2_stopped.log`).
3. Run 3 (resume, 2026-10-23 11:00–11:33 IST, log `logs/robust/p5_composites.log`) kept the 89 existing files and wrote the
   remaining 37; exit code 0. Its draw lines equal run 1's for all 87 and run 2's for all 89 (diff checked).
- **Limitation:** the 89 resumed files were verified by their cloud mask (equal to the redrawn window), shapes, dtype and value
  range. Their optical bands were not compared with an independent re-read, and run 1 recorded no per-file hash before the
  interruption; no byte-for-byte verification is claimed. Every file's sha256 is recorded in the manifest and below.

**Detector inputs.** 40 validation (4 clean + 36 composites) and 100 test (10 clean + 90 composites) files. The first export
attempt for each split stopped at its first composite (logs `p5_export_{val,test}_run1_failed.log`): the check that a
composite's mask fraction equals the manifest compared a float32 torch mean with the float64 manifest value at 1e-12 tolerance
(difference about 3e-9). All 126 fractions are exactly equal in float64. The check was changed to compare in float64, exactly
(as the builder computes it), the one clean file each attempt had written (`beihai_clean`, `brasilia_clean`; no model involved)
was deleted, and both splits were exported again in full.

**Detector, validation only.** s2cloudless 1.7.3 (`.venv_cloud`, packages in `detector_env_freeze.txt`), model
`pixel_s2_cloud_detector_lightGBM_v0.1.txt`, sha256 `0f92b6822cc0ad42982f55bab9b1688329e119451ad9a9e0a56384b9f37bc61d`; library
defaults threshold 0.4, average_over 1, dilation_size 1, all_bands false. Run on the 40 validation inputs
(`logs/robust/p5_detect_val.log`). `detector_outputs/test/` does not exist.

**Checkpoints.** The sha256 of each of the 30 checkpoints equals its entry in `ARTIFACT_SHA256_2026-10-22_post4b.txt` (checked
when this file was written); concatenated hashes per arm are below.

| file | sha256 |
|---|---|
| `build_phase5_composites.py` | `dc68252aca9cbd2c683e86b83e0402406b575b5f6083dd075f7effee2334c214` |
| `export_phase5_detector_inputs.py` | `93aa703422621bbbe4c54a6ecc5e8a41201bce9916b5a8671150934e19597d78` |
| `detect_phase5.py` | `db74ba14372581195ed01147b64f35d9c58f975b920c2859a96ea2d639101d0c` |
| `select_phase5_operating_points.py` | `ab19fef4d1964244394f5623691d26432a8b6222482b356fb1adb065e17e380f` |
| `smoke_phase5.py` | `f4a5157a2dd0506dea9ed1c8b569a91bfc21edb02edca129dd21573d335cbdc7` |
| `eval_phase5.py` | `823e49f747ec0aaa2d7a97000378b9d8ee0f8931d7b44da39f22c3ae132e0ecd` |
| `analyze_phase5.py` | `f019323187cb1bbaafa3fa79529fa3fceeb8a24f970d26ab319db8dda8e6d714` |
| `audit_p5.py` | `9a7c08b29aac9d622dfeb18427cd3017d43c7487c318cba46adf8b6d54259a43` |
| `replay_spatial_mask_bank_sources.py` | `e38239749dda971587cde55340c258af57648b2119d928b7f4b90fa06d3e0e1d` |
| `src/phase5/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `src/phase5/composite.py` | `33fcd2875289a412cba2c0ee6449ea8105f8b5b12edaa55d8d033ebd34328fe6` |
| `eval_phase4a.py` | `5d831a81126b97c99778fa835ec35ef068520b30b960756c3c3b97b3ccc70f62` |
| `analyze_phase3.py` | `96bd2cc6811321ad00438f45cb81d3e01cc485d4406c2aa5fc7e770a6354a04b` |
| `src/phase3/corruption.py` | `6418fde31779dd219f3b2ec7204c54760b2f7b7d70bb9070db86012639fb7d1e` |
| `src/phase3/remote_zip.py` | `89292d00eace9d3d374ff2e9cd5ab32a76b23cd42da4d9faf879ea939ccf8448` |
| `src/phase3/mask_errors.py` | `7758b8cc436c83de02784a39adaf8725b15fe60b6a2437d0b2429f8807f667bd` |
| `src/training/train_robust.py` | `c7d812e3fd8dcf2c9bd8c7f4d3faeb1bee0c97fc256f83a76ee2194e15a6436c` |
| `src/training/train_sar_baseline.py` | `bbd0044580a207a318b452f6c153d3c67b94ea4cc1f388283f8012fecaf19ec8` |
| `src/datasets/oscd_dataset.py` | `bc9b64bc2fb9808efb2d8b6767a594f19385e52f4514c09633b3a7af8486fc43` |
| `verify_prereg_hashes.py` | `7455f73c1ea05f75673583d8b939ea9af38616685c42a5eaefe372b0e85c1c4d` |
| `phase5/composite_manifest.json` | `b98fc7c28c43b7e93602b4d129aee4ca4c602590f244caa8ab8411a30c4b01b8` |
| `phase5/detector_outputs/val/detector_info.json` | `117ad9bbee4e229626c7906507c04a397ec324c725bc07dfbdc5efa4ebcbee2c` |
| `phase5/detector_env_freeze.txt` | `7f705a9926ce9801ff1274fcb20680b8109f864c0108faf98a4d4d3c934524fd` |
| `phase4b/spatial_mask_bank_sources.json` | `2f45d11bd69684e914b18c27ba0afcff5548ae840fd97957fe6277dd349fad47` |
| `phase4b/spatial_mask_bank.json` | `61c945dad58b3d3e9515dad8c236a6faf5c2318f7bbfe8bf4a58c56ad06d0918` |
| `phase3/mask_manifest.json` | `e1c60c62db93777a11e1d049d8266b8b5c4577449346124da7ad6f5f404b2b87` |
| `ARTIFACT_SHA256_2026-10-22_post4b.txt` | `5a95e9acfd3dfba9b18923e3b2f842d0f99b90073e385e649e0e1f713ecd7a24` |
| `datasets/s2_cloud_mask_catalogue/classification_tags.csv` | `46b88e07480f36a4d2fa27f01a1ee9987b1def7ad08f1eb721a88ef0b5288d25` |
| `logs/robust/p5_composites_run1_hung.log` | `abadf6766df68e4adfe4963d6bca7b3be0e2bb24813a77996ca1628fb2b648b7` |
| `logs/robust/p5_composites_run2_stopped.log` | `f8c1ef464f2c8a494e377b420409d185d739474bc4cbb45474ba68771636a6da` |
| `logs/robust/p5_composites.log` | `047c4efda8fc96632d7fbf1b6c3fa56511838effbbd9f17ca8490be4e1e8a115` |
| `logs/robust/p5_bank_replay.log` | `c944e7d9e004a3aac63e0b80859db7e1f72140bc3ab6433b931c9b595aefdd5d` |
| `logs/robust/p5_export_val_run1_failed.log` | `67451850bc9bc25b4db52069249fa5f2da23809f8aab3069e43587a55781642b` |
| `logs/robust/p5_export_test_run1_failed.log` | `67451850bc9bc25b4db52069249fa5f2da23809f8aab3069e43587a55781642b` |
| `logs/robust/p5_export_val.log` | `0b0f78406ff74b5481563234d30b435e26397310e9b43aa69906592e54cdd5f6` |
| `logs/robust/p5_export_test.log` | `f3a0fa3b3136324e4ea80be753715ab7d4dcbc9f269219060f60a0f8646112af` |
| `logs/robust/p5_detect_val.log` | `90c8a1b5ae853d8f970a49869319a3cfc5ebe2b8c6b871ffce088377cd9797e4` |
| `phase5/composites/*.npz (126, concatenated, sorted)` | `7d61b39c782b76419518a39444b5174eb98125e15cab93d9f77434303ebf37fe` |
| `phase5/detector_inputs/val/*.npz (40, concatenated, sorted)` | `89cb50d647364a56ae88b9ab425e59bbbd8d985e68191bb374e98db0bdfe9c8b` |
| `phase5/detector_inputs/test/*.npz (100, concatenated, sorted)` | `e77cfe6b3ee906cbd74c835283d112b314c667b04ec25b9067f36a3d27270122` |
| `phase5/detector_outputs/val/*.npz (40, concatenated, sorted)` | `924f92b3b9f06c981442adc021c58b677eef09e7be83b4e90cdcc5fb8169240f` |
| `checkpoints/robust/p2_full_seed*.pt (10, concatenated, sorted)` | `5817ec427596daa2062c84902eed54e727681457e33aae4d4129b59c504b844e` |
| `checkpoints/robust/p2_dropout_seed*.pt (10, concatenated, sorted)` | `8bc2646ed372b5e1c60f9a0d17f20b7735064d94d54d284bbed785e7a13f40e2` |
| `checkpoints/robust/p4b_spatial_seed*.pt (10, concatenated, sorted)` | `50bea63094bd48639e864a9e83eca9f189809eba2927e9893e6a206df62dd438` |
| `phase4abis/runs/*.npz (20, concatenated, sorted)` | `de32beea2f68fb2e088fb13e95867bd0bca8e265d07765859d157dfcc72c0286` |
| `phase4b/runs/*.npz (10, concatenated, sorted)` | `e53ca32cf1b25d129b8729181e067e197d1724a2723a3a97c52acb6143e41f13` |

## 12. Pre-evaluation verification (to be appended after this file is committed, before any test evaluation)

To be appended: operating-point selection result (validation only), smoke-test result (validation only), test detector run, and
the hashes of `operating_points.json`, `detector_outputs/test/detector_info.json` and `smoke_phase5.py`.

### §12 appended 2026-10-23 (after commit `d9e29ff`, before any test evaluation of a change model)

**Operating points (validation composites only; `select_phase5_operating_points.py`, log `logs/robust/p5_select.log`).**
Pooled over the 36 validation composites against M:

| point | t | r | cloud recall | precision | mask F1 | clear pixels masked | target reached |
|---|---|---|---|---|---|---|---|
| rb95 (primary) | 0.25 | 8 | 0.954 | 0.502 | 0.658 | 0.458 | yes |
| rb90 | 0.55 | 12 | 0.902 | 0.655 | 0.759 | 0.230 | yes |
| rb99 | 0.05 | 16 | 0.986 | 0.325 | 0.489 | 0.993 | **no**: highest-recall fallback (§5), flagged |
| bal | 0.65 | 6 | 0.791 | 0.887 | 0.836 | 0.049 | — |

Noted before any test evaluation: rb95 reaches its recall target only by masking 46% of clear validation pixels; rb99 does not
reach 0.99 at any grid point, and its fallback is at the grid boundary (t 0.05, r 16) and masks almost every clear pixel. Both
follow the registered rule; neither is a protocol change. rb99 is a secondary condition.

**Smoke test (validation only; `smoke_phase5.py`, log `logs/robust/p5_smoke.log`):** **38 checks, 0 failed**: pools reproduce the
manifest and are disjoint; composite files match the manifest; composite construction (SAR and outside-M unchanged, inside-M equal
to the catalogue bands); detector-input band order; detector outputs and library defaults; detector-mask rule (dilation equals a
Euclidean disk); rb95 and bal follow the selection rule; operating points used validation composites only; and for
p2_dropout_seed1 and p4b_spatial_seed1 on the validation cities: clean counts equal the stored counts, clean argmax equals the stored
threshold, inside + outside M equals pooled, outside M unchanged, deterministic, eval mode. `smoke_phase5.py` is unchanged since
`d9e29ff`.

**Test detector (`detect_phase5.py test`, log `logs/robust/p5_detect_test.log`):** 100 files (10 clean + 90 composites); detector
configuration identical to the validation run (same library version, defaults and model sha256). No change model was run.

| file | sha256 |
|---|---|
| `phase5/operating_points.json` | `ae0ec54611d0a076b8f6474fd1bf088cdc51b4127ddfb812273542458d582619` |
| `phase5/detector_outputs/test/detector_info.json` | `c92d58dcf34040ce5644b1434687cebb493964756b66499a6d33d572f42c4957` |
| `phase5/detector_outputs/test/*.npz (100, concatenated, sorted)` | `357bf292a41021dac1391fa35d91f6b5dc8657063a1b5a7384a817a9bfd305c1` |
| `smoke_phase5.py` | `f4a5157a2dd0506dea9ed1c8b569a91bfc21edb02edca129dd21573d335cbdc7` |
| `phase5/smoke/smoke_results.json` | `9cbe6c6d7dfccb4b4980cef9d702f3ce0a2fed7ca89294a8f4c032d5950868fa` |
| `logs/robust/p5_select.log` | `2c5dc90aff6de9adb4894ba384ec905c2cf813c72615a3e12b315157ef0c020f` |
| `logs/robust/p5_smoke.log` | `75af0e3bea4efe327c2d0751adb67ebfaffacf75db67a4e04e03417eaf72d0ec` |
| `logs/robust/p5_detect_test.log` | `d6bf6fdf9b2796586bced0faf6df6a98cdfe3ca10339c36f7456ed94ebc5d0fb` |
