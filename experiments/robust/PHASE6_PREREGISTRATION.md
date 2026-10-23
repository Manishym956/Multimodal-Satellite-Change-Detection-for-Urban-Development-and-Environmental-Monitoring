# Step 6 pre-registration: baselines and the asynchronous configuration (RRD RQ4, RQ1)

Written 2026-10-23, before any Step 6 arm was trained (no `experiments/robust/p6_*` run exists; only the parity run
`parity_rrd_seed1` and the validation-only smoke test). Base commit `f56ba71`, branch `robust-phase4`. Part of the plan in
`docs/final/FINAL_RESEARCH_PLAN.md`. Any later change is reported as a deviation.

## 1. Questions and hypotheses

The final paper answers the Research Requirement Document's RQ1–RQ5. The robustness study (Phases 2–5) compared only
DS-UNet arms, and the companion paper's baselines used a different protocol (different model code, BCE loss, lr 1e-4,
no augmentation, threshold 0.5, 3–5 seeds), so they cannot enter a comparison with A and B.

- **RQ4 (primary):** does the dual-stream DS-UNet (A, full-only) differ from a SAR-only, an optical-only and an
  input-fusion U-Net on full input? Two-sided.
- **RQ1 (primary):** without the post-change optical image, does SAR + historical (pre-change) optical differ from SAR
  alone? Tested for B (modality dropout) and for an asynchronous DS-UNet trained only on SAR + historical optical.
  Two-sided.
- Known before writing: Phase 2 B `no_o2` 0.431 and `no_opt` 0.440 (not a registered contrast, not evidence either way);
  the companion paper's async 0.309 vs SAR-only 0.241 under its own protocol (not significant after Holm).

## 2. Arms (all official DS-UNet code, `repos/2/DS_UNet@8bfa4be`; SAR = Hafner VV rasters)

| arm | model (`src/training/train_rrd.py --arch`) | training conditions | status |
|---|---|---|---|
| `p2_full` (A) | DualStreamUNet | full=1 | existing (Phase 2, `train_robust.py`) |
| `p2_dropout` (B) | DualStreamUNet | full=0.5, no_o2=0.25, no_opt=0.25 | existing |
| `p6_unet_sar` | UNet on VV t1, t2 (2 channels) | full=1 | new |
| `p6_unet_opt` | UNet on S2 t1, t2 (26 channels) | full=1 | new |
| `p6_unet_fusion` | UNet on VV + S2 stacked at the input (28 channels), DS-UNet's "fusion" mode | full=1 | new |
| `p6_async` | DualStreamUNet | no_o2=1 | new |

Recipe identical to Phase 2: `--steps 2500 --eval-every 25`, Jaccard-like loss, Adam 5e-5, weight decay 5e-4, batch 64,
32×32 importance crops, flips and rotations, seeds 1–10, split train 10 / validation 4 / test 10. Checkpoint: highest
mean validation F1 over the trained conditions; per-condition threshold: validation-F1 argmax. As in Phase 2 and Step 4B,
the training script evaluates the selected checkpoint on the test cities once at the end of each run and prints it; no
setting is changed after any test output.

**Parity (done):** `train_rrd.py --arch dsunet --sar hafner_vv` with B's mixture reproduces `p2_dropout_seed1` exactly
(`logs/robust/p6_parity.log`: weights, validation history, selection and stored test counts identical). The test F1 it
printed is Phase 2's already-reported seed-1 value.

## 3. Analysis (`analyze_phase6.py`, fixed by hash below)

Test F1 at each run's validation-selected threshold for the condition (Phase 2's rule). Paired over seeds 1–10; paired
t-test, Holm within each family; mean Δ, 95% CI, d_z, seeds with Δ > 0, Wilcoxon and Holm-Wilcoxon alongside.
Labels: "<X> better" / "<X> worse" if Holm p < 0.05 with that sign, else "no detectable difference".

- **primary_rq4 (Holm over 3):** full input: A − SAR-only, A − optical-only, A − input fusion. X = DS-UNet.
- **primary_rq1 (Holm over 2):** B@no_o2 − SAR-only@full; async@no_o2 − SAR-only@full. X = SAR + historical optical.
- **secondary_fusion (own family):** fusion − optical-only, fusion − SAR-only, optical-only − SAR-only (full input).
- **secondary_async (own family):** async@no_o2 − B@no_o2; B@no_o2 − B@no_opt.
- **Descriptive:** F1, precision, recall, IoU, false-alarm rate for every arm and condition, at the selected threshold and
  at 0.5. An undefined rate (zero denominator) counts as 0.
- **Sensitivity:** with 10 pairs and power 0.8, the minimum detectable effect is 1.00 SD of the paired differences at
  α = 0.05, 1.22 SD at α = 0.05/3 and 1.14 SD at α = 0.05/2 (first Holm steps; noncentral t).

`audit_p6.py` (written before any Step 6 run) recomputes every rate from raw confusion counts and every family with
numpy/scipy, and checks the registered settings, cities, selection rule and source hashes.

## 4. Interpretation guidance (pre-specified; not tests)

- RQ4: "DS-UNet better" against a baseline supports modality-specific processing over that baseline under this protocol.
- RQ1: "SAR + historical optical better" for B or async supports RQ1; "no detectable difference" is reported as no
  measurable benefit of historical optical over SAR alone under this protocol; "worse" is reported as harm.
- Results concern one benchmark (OSCD, 10 test cities), VV only, these architectures and this recipe.

## 5. Stopping rules and failures

The gate script `logs/robust/p6_run.sh` stops if the parity line is missing, the smoke test has a failure, or any hash
below has changed. A run that does not finish stops the chain; it is reported and re-run from scratch with the same seed
(training is deterministic). No run is excluded.

## 6. Smoke test (validation only)

`smoke_phase6.py` → `experiments/robust/phase6/smoke/smoke_results.json`, log `logs/robust/p6_smoke.log`: **18 checks,
0 failed**. Each new architecture trained 20 steps and was evaluated on the validation cities only; the analysis and
audit ran end to end on stand-in data (copies of Phase 2's already-reported metrics under the Step 6 arm names, in a
temporary folder): of the audit's 1042 checks, all statistical checks agreed with the analysis, and the only 80 failures
were the expected "checkpoint exists" and "registered settings" checks, which need real Step 6 runs. The smoke test exposed one audit
bug (scipy's Wilcoxon raises on all-zero differences), fixed before this file was written.

## 7. Hashes (sha256)

| file | sha256 |
|---|---|
| `src/training/train_rrd.py` | `cf918eb7481f8cd98352a4702f16913f03f373cb9d5cd077d9fe6f4b0adcbb32` |
| `src/training/train_robust.py` | `c7d812e3fd8dcf2c9bd8c7f4d3faeb1bee0c97fc256f83a76ee2194e15a6436c` |
| `src/training/train_sar_baseline.py` | `bbd0044580a207a318b452f6c153d3c67b94ea4cc1f388283f8012fecaf19ec8` |
| `src/datasets/oscd_dataset.py` | `bc9b64bc2fb9808efb2d8b6767a594f19385e52f4514c09633b3a7af8486fc43` |
| `src/evaluation/metrics.py` | `6d49b1892b9a2689bc8c4c23673535b06e5dc4f45df86c1bd3c0279a9f7b4854` |
| `repos/2/DS_UNet/utils/networks.py` | `a3e7216b6054fe85cb3797233b08aa08f25b4cfb1d70c2b159e00e1bb3066830` |
| `repos/2/DS_UNet/utils/loss_functions.py` | `6e71a2e830f9c39886aa114d2dc23d763e3fe653e46deae41c07f51a4f204a7f` |
| `analyze_phase6.py` | `f491a298f98a00fde88f99a8b8eba6242ff29063b1a9640df4b25a0d6656b841` |
| `analyze_phase3.py` | `96bd2cc6811321ad00438f45cb81d3e01cc485d4406c2aa5fc7e770a6354a04b` |
| `audit_p6.py` | `dd7856c57c87c3b60bb59315696afacf94ddb43bfb4b33682de6964cc0c246f5` |
| `smoke_phase6.py` | `583ea2379f28ff5bed034c3b641dc5f1b109fe5ddc7024e1202417bbd6a1b38e` |
| `verify_prereg_hashes.py` | `7455f73c1ea05f75673583d8b939ea9af38616685c42a5eaefe372b0e85c1c4d` |
| `logs/robust/p6_run.sh` | `53c9cf2a2786feb716c5627c57b7dae7389b05fa7d7f42585ee5c737561e9bb1` |
| `logs/robust/p6_parity.sh` | `80902991f3036d525cd8541e02c25b052449c1464c3ef9a209b7539f65ba9e8a` |
| `logs/robust/p6_parity.log` | `717ee8a94804f4b9e3f76e0f3a5b56d495d5d3f07600a57533e12f9430f76587` |
| `logs/robust/p6_smoke.log` | `d4dd0034abcc141e693adf86343b0953c45f863aaed082424790e21a8356dc38` |
| `phase6/smoke/smoke_results.json` | `ca56e71243ff06034fc183a2d558ba5691ffd10243b1892efee90b63231e190b` |
| `p2_full_seed*/metrics.json (10, concatenated, sorted)` | `856a67d41a7e20d9f6690551a9f6322eab38f9a8dc823d56390e30a110963dd1` |
| `p2_dropout_seed*/metrics.json (10, concatenated, sorted)` | `e2af8d130d433b912df9710fedc68927c4edb4a9a9a718b445ac1ff4d31e6f70` |
| `../../checkpoints/robust/p2_full_seed*.pt (10, concatenated, sorted)` | `5817ec427596daa2062c84902eed54e727681457e33aae4d4129b59c504b844e` |
| `../../checkpoints/robust/p2_dropout_seed*.pt (10, concatenated, sorted)` | `8bc2646ed372b5e1c60f9a0d17f20b7735064d94d54d284bbed785e7a13f40e2` |
| `../../docs/final/FINAL_RESEARCH_PLAN.md` | `770163f6294825ce28a6325aa72f5cb2a4c5b8c61e6ebcf52db51397c9278f1c` |
