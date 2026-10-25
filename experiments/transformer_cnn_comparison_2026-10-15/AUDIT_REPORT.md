# Audit: Transformer tuning and CNN benchmark comparison

**Audit scope:** read-only review of the 2026-10-15 Transformer tuning package, original OSCD and Scenario 2 experiments, existing CNN runs/checkpoints, split definitions, and the release artifact. Existing test results have already been viewed in prior work; this report does not claim test-set secrecy was preserved historically.

## Existing experiment inventory and verified values

The previous Transformer campaign contains six valid seed-42 candidate runs (two optimizer settings for each of OSCD self-attention, OSCD cross-attention, and Scenario 2 cross-attention), six selected-setting repeats at seeds 43/44, and five preserved failed/invalidated attempts. Validation selection and threshold selection were used as documented in `experiments/transformer_tuning_2026-10-15/TUNING_REPORT.md`. The selection stage used validation pooled F1 at .5; threshold calibration used validation only. The prior audit reports 24 final comparison rows, 6 repeatability rows, and 17 registered run directories.

The original OSCD metrics files verify the following pooled test counts at threshold .5:

| Baseline | F1 | TP / TN / FP / FN | Loss / pos_weight | Epoch selected | Training time |
|---|---:|---|---|---:|---:|
| SAR-only U-Net | .3558 | 46,407 / 2,863,458 / 55,401 / 112,670 | BCE / 1 | 21 | recorded in metrics |
| Optical-only U-Net | .2663 | 45,025 / 2,784,846 / 134,013 / 114,052 | BCE / 1 | 50 | recorded in metrics |
| Early-fusion U-Net | .3520 | 72,104 / 2,740,370 / 178,489 / 86,973 | BCE / 1 | 46 | not recorded |
| Synchronous dual-stream CNN | .3868 | 86,137 / 2,718,735 / 200,124 / 72,940 | BCE / 1 | 31 | 2,993.53 s |
| Naive asynchronous dual-stream CNN | .2052 | 28,849 / 2,825,617 / 93,242 / 130,228 | BCE / 1 | 46 | 616.40 s |
| Clean asynchronous dual-stream CNN | .4117 | 57,764 / 2,855,097 / 63,762 / 101,313 | BCE / 1 | 26 | 387.12 s |
| Self-attention Transformer | .1615 | 75,307 / 2,220,865 / 697,994 / 83,770 | weighted BCE / 41 | 37 | 341.39 s |
| Cross-attention Transformer | .3564 saved; .3564 rerun | 88,959 / 2,667,663 / 251,196 / 70,118 saved; 88,965 / 2,667,639 / 251,220 / 70,112 rerun | weighted BCE / 41 | 33 | 334.17 s |

Transformer tuned seed-42 test F1 at .5 is .2357 (self) and .2477 (cross). Their best validation-selected threshold test F1 values are .2253 at .4375 and .2594 at .525, respectively. Scenario 2 at .5 is: asynchronous CNN .079446 (TP/FP/TN/FN 1,601/31,844/915,825/5,258), synchronous CNN .110181 (3,896/59,965/887,704/2,963), and cross-attention Transformer .041176 (3,913/179,292/768,377/2,946). These Scenario 2 scores are building-probability XOR proxy results, not general environmental-change accuracy.

## Comparability audit

- OSCD models use the same 10/4/10 city split (seed 42), full-city evaluation, first listed SAR orbit for evaluation, and pooled pixel metrics. Their input protocols differ by task: synchronous CNN uses SAR plus optical at both dates (28 channels); clean asynchronous CNN and Transformers use two SAR channels plus the 13-band historical optical input (15 channels). The naive asynchronous model includes zero-filled missing optical channels and is an ablation, not the clean protocol.
- CNN runs use unweighted BCE; OSCD Transformers use `pos_weight=41`. Training data sampling, model size, and tuning budget also differ. Therefore the direct table is a descriptive benchmark, not an isolated architecture effect. The clean asynchronous CNN has 27,276,609 parameters; Transformer models have about 1.16–1.36M. The synchronous dual-stream CNN took about 50 minutes for one 50-epoch baseline run, while the clean asynchronous CNN took about 6.5 minutes.
- No existing OSCD CNN probability-map artifacts were found. Original metrics contain only fixed-threshold counts. This package will generate a validation-only threshold sweep and a single cached inference set per CNN for the validation-calibrated test view; it will not retrain CNNs. Given the large runtime mismatch and existing benchmark gap, a matched CNN tuning campaign is deferred and these CNNs remain fixed baselines.
- Existing successful Transformer seeds 43/44 are already present and will be reused. No Transformer training will be repeated. There are no OSCD CNN repeat seeds in the inspected baseline package. Scenario 2 has prior synchronous CNN seeds 43/44, but their calibration uses a different valid-pixel protocol and is not pooled with the current all-pixel baseline comparison.
- Existing training code uses `model.eval()` and inference mode for validation. The previous tuning driver's initial validation-mode defect is documented; all affected attempts were excluded and the corrected valid runs were used. New code in this directory will run validation in eval/inference mode and will not modify source loaders/models.

## Missing evidence and limitations

No result here establishes temporal-gap sensitivity. Scenario 2 source archive integrity and `s2`/`masked`/bad-index semantics remain unresolved; the current fixed sample set is provisional. Its test set contains four sites. OSCD has 10 test cities, but pixel-pooled scores still do not make pixels independent geographic samples. The validation set has four cities, so threshold calibration is potentially unstable. Existing Scenario 2 cross baseline rerun differs from its saved counts by a few pixels; preserve both values and label their source.

The prior release at `artifacts/final_release_2026-10-15/` is read-only for this task. New outputs, checks, and plots belong only in this directory.
