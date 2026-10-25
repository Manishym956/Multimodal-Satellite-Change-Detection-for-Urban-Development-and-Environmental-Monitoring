# Transformer Tuning vs CNN Benchmarks — Final Report

**Experiment directory:** `experiments/transformer_cnn_comparison_2026-10-15/`  
**Scope:** additive audit and comparison using existing trained checkpoints; CNN operating thresholds calibrated from validation only. No models were retrained and no original artifacts were modified.

## Findings

The tuned Transformers did not beat either the synchronous dual-stream or clean asynchronous CNN on OSCD at the shared .5 threshold. The clean asynchronous CNN scored .4117 F1, versus .2357 for tuned self-attention and .2477 for tuned cross-attention. Synchronous dual-stream CNN scored .3868. Even the original, untuned cross-attention baseline (.3564 saved F1) remained below both CNNs. The tuned Transformers did exceed some weaker CNN baselines: seed-42 self and cross attention both beat the naive asynchronous CNN (.2052), and tuned cross-attention exceeded the SAR-only CNN (.3558) only before tuning, not after.

| OSCD model and status | Threshold policy | Test F1 | Precision | Recall | IoU | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|
| Synchronous dual-stream CNN, fixed | .5 | .3868 | .3009 | .5415 | .2398 | 200,124 | 72,940 |
| Clean asynchronous CNN, fixed | .5 | .4117 | .4753 | .3631 | .2592 | 63,762 | 101,313 |
| Self-attention Transformer, original | .5 | .1615 | .0974 | .4734 | .0879 | 697,994 | 83,770 |
| Self-attention Transformer, tuned seed 42 | .5 | .2357 | .1428 | .6750 | .1336 | 644,509 | 51,702 |
| Cross-attention Transformer, original saved metrics | .5 | .3564 | .2615 | .5592 | .2168 | 251,196 | 70,118 |
| Cross-attention Transformer, tuned seed 42 | .5 | .2477 | .1578 | .5755 | .1414 | 488,537 | 67,522 |

Tuning’s largest verified seed-42 gain was OSCD self-attention: **+0.0742 F1 (+45.9%)** at .5. Scenario 2 cross-attention improved from .0412 to .0850 at .5, but remains below the fixed synchronous CNN (.1102). On Scenario 2, tuned cross-attention at .5 did slightly exceed the fixed asynchronous CNN (.0794); its validation-selected threshold score (.0908) also exceeded the existing asynchronous CNN’s calibrated score (.0807). It did not exceed the synchronous CNN, whose calibrated score was .1994. These results concern the existing building-probability XOR proxy, not general environmental change.

## Validation-selected operating points

Thresholds were selected by pooled validation F1. The two OSCD CNN thresholds were newly calibrated from one cached inference pass per city; the four-city validation set makes these thresholds unstable candidates, not universal operating points. Their respective test F1s were **.3131** for clean asynchronous CNN at .20 and **.3401** for synchronous dual-stream CNN at .4375. The Transformer selected seed-42 thresholds were .4375 (self) and .525 (cross), yielding test F1s .2253 and .2594. In the three-seed Transformer view, seed-43 tuned cross-attention at its validation-picked .575 threshold scored .3267, above the clean CNN’s calibrated .3131 but below the synchronous CNN’s .3401. It was still below the clean CNN’s .4117 at the common .5 threshold and is not a robust across-seed win.

## Repeatability and fairness

For tuned Transformers at common .5, pooled test F1 mean ± sample SD over seeds 42/43/44 was **.2115 ± .0244** (OSCD self), **.2307 ± .0647** (OSCD cross), and **.0731 ± .0104** (Scenario 2 cross). The tuned cross model’s seed variability is substantial. CNN baselines have only seed 42 in OSCD, so no seed-level comparison of variability against those CNNs is possible. Scenario 2 has standard synchronous CNN outputs for seeds 42/43/44 in the all-pixel protocol; their F1 mean is about .1252 with sample SD .0130. Masked-loss repeats are not pooled with standard loss.

The OSCD comparisons are not architecture-isolated. CNN baselines use unweighted BCE, while Transformers used class-weighted BCE with `pos_weight=41`; the clean asynchronous CNN has about 27.3M parameters, compared with about 1.16–1.36M for Transformers. The synchronous CNN sees both optical dates (28 channels), whereas clean asynchronous CNN and Transformers use historical optical only (15 channels). Existing CNNs were not tuned in this package and are marked fixed baselines. A matched CNN search is deferred: the synchronous baseline alone took 2,993.53 seconds for one 50-epoch run, and the original CNN scripts automatically evaluate test data on completion and write to fixed baseline paths. A safe comparable search would first require an isolated, no-test model-selection runner and substantially more compute. Thus score comparisons are informative benchmarks, not causal evidence that attention alone explains performance.

Validation threshold choice can reverse apparent gains: the clean asynchronous CNN’s .20 validation-picked threshold fell from .4117 at .5 to .3131 test F1; the synchronous CNN’s .4375 threshold fell from .3868 to .3401. Do not present validation calibration as test improvement or claim the threshold policy is generally superior.

## Data and inference limits

The OSCD split is the existing 10/4/10 city split; all model comparisons use the same 10 test cities. No temporal-gap claim is supported: OSCD SAR dates are not precise enough for that analysis. Scenario 2 uses a 20/4/4 site split with locked SHA-256 `5d0841adf0c651166cb18ecea310c6ef8cce69172065fc2d699a29083c272e27`; its archive checksum and `s2`/`masked`/bad-index meanings remain unresolved, the test set is only four sites, and eligibility was not expanded. The all-gap common-site intersection remains seven (5 train, 1 validation, 1 test), so no controlled temporal-gap sweep is justified.

The original OSCD cross-attention checkpoint rerun differs slightly from its saved counts: rerun TP/TN/FP/FN `88,965 / 2,667,639 / 251,220 / 70,112`, compared with saved `88,959 / 2,667,663 / 251,196 / 70,118`. The saved F1 was used for the fixed .5 baseline row; the rerun and threshold-calibrated rows are retained separately in the before/after table. This changes F1 by only about .0000026, but the difference is not silently reconciled.

## Outputs and reproduction

- `AUDIT_REPORT.md` and `EXPERIMENT_PROTOCOL.md`: inventory, comparability assessment, and frozen protocol.
- `BEFORE_AFTER_RESULTS.csv`: all original/tuned Transformer test rows, both threshold policies, seeds and metric provenance.
- `CNN_TRANSFORMER_COMPARISON.csv`: OSCD and Scenario 2 fixed-threshold benchmarks plus validation-selected comparisons.
- `AGGREGATE_RESULTS.csv`: per-cohort means and sample SDs with run counts; one-run CNN OSCD baseline SDs are `N/A`.
- `TRAINING_RUNS.csv`: original baselines and prior tuning runs, including failed/invalidated attempts.
- `cnn_validation_threshold_sweeps.csv`, `cnn_validation_selected_thresholds.json`, `cnn_frozen_test_thresholds.csv`: new read-only CNN calibration outputs.
- Plots: `before_after_tuning.png`, `cnn_transformer_comparison.png`, `per_seed_f1_variation.png`, `validation_precision_recall.png`.
- `validation_checks.json` and `sha256_manifest.json`: integrity and metric checks plus hashes for outputs and referenced checkpoints.

Reproduce table/plot generation with:

```powershell
.\.venv\Scripts\python.exe experiments\transformer_cnn_comparison_2026-10-15\calibrate_oscd_cnn_thresholds.py
.\.venv\Scripts\python.exe experiments\transformer_cnn_comparison_2026-10-15\build_comparison_outputs.py
.\.venv\Scripts\python.exe experiments\transformer_cnn_comparison_2026-10-15\create_sha256_manifest.py
.\.venv\Scripts\python.exe experiments\transformer_cnn_comparison_2026-10-15\validate_comparison.py
.\.venv\Scripts\python.exe experiments\transformer_cnn_comparison_2026-10-15\create_sha256_manifest.py
.\.venv\Scripts\python.exe experiments\transformer_cnn_comparison_2026-10-15\validate_comparison.py
```

The prior Transformer training campaign is documented and reproducible from its run configs and `experiments/transformer_tuning_2026-10-15/tuning_pipeline.py`. Do not rerun its test stage for a new selection. The existing `artifacts/final_release_2026-10-15/` and all original baseline artifacts remain untouched.
