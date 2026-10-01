# Baseline comparison

Three completed runs are compared here. The numbers are the recorded validation and test results. This note does not rank the models.

Pooled test metrics combine pixel counts across the 10 official test cities. City means are unweighted. Sample standard deviation uses divisor n−1. An undefined rate is left undefined and is omitted from that metric's mean and standard deviation.

FAR = FP / (FP + TN). The decision threshold is 0.5 after sigmoid.

## Pooled metrics

| Model | Best validation epoch | Best validation F1 | Precision | Recall | F1 | IoU | FAR | TP | FP | FN | TN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SAR-only | 21 | 0.0662 | 0.4558 | 0.2917 | 0.3558 | 0.2164 | 0.0190 | 46407 | 55401 | 112670 | 2863458 |
| Optical-only | 50 | 0.2387 | 0.2515 | 0.2830 | 0.2663 | 0.1536 | 0.0459 | 45025 | 134013 | 114052 | 2784846 |
| Synchronous SAR + Optical Fusion | 46 | 0.2538 | 0.2877 | 0.4533 | 0.3520 | 0.2136 | 0.0612 | 72104 | 178489 | 86973 | 2740370 |

## City-level F1

| City | SAR-only F1 | Optical-only F1 | Fusion F1 | Fusion F1 − SAR-only F1 |
| --- | ---: | ---: | ---: | ---: |
| brasilia | 0.0651 | 0.1032 | 0.2920 | 0.2269 |
| montpellier | 0.2405 | 0.6492 | 0.6276 | 0.3871 |
| norcia | 0.3375 | 0.1657 | 0.3160 | -0.0216 |
| rio | 0.2550 | 0.3148 | 0.5259 | 0.2710 |
| saclay_w | 0.3960 | 0.0836 | 0.1511 | -0.2448 |
| valencia | undefined | 0.0150 | 0.0217 | undefined |
| dubai | 0.4630 | 0.1203 | 0.1768 | -0.2862 |
| lasvegas | 0.3032 | 0.5215 | 0.4322 | 0.1291 |
| milano | 0.2255 | 0.0122 | 0.1956 | -0.0299 |
| chongqing | 0.3261 | 0.0900 | 0.3415 | 0.0154 |

## City-level metrics

| City | Model | Precision | Recall | F1 | IoU | FAR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| brasilia | SAR-only | 0.4195 | 0.0353 | 0.0651 | 0.0336 | 0.0013 |
| brasilia | Optical-only | 0.0556 | 0.7072 | 0.1032 | 0.0544 | 0.3180 |
| brasilia | Synchronous SAR + Optical Fusion | 0.1905 | 0.6253 | 0.2920 | 0.1710 | 0.0704 |
| montpellier | SAR-only | 0.5134 | 0.1570 | 0.2405 | 0.1367 | 0.0109 |
| montpellier | Optical-only | 0.6912 | 0.6119 | 0.6492 | 0.4806 | 0.0199 |
| montpellier | Synchronous SAR + Optical Fusion | 0.5474 | 0.7353 | 0.6276 | 0.4573 | 0.0443 |
| norcia | SAR-only | 0.5186 | 0.2502 | 0.3375 | 0.2030 | 0.0031 |
| norcia | Optical-only | 0.1046 | 0.3985 | 0.1657 | 0.0904 | 0.0457 |
| norcia | Synchronous SAR + Optical Fusion | 0.2794 | 0.3635 | 0.3160 | 0.1876 | 0.0126 |
| rio | SAR-only | 0.3803 | 0.1918 | 0.2550 | 0.1461 | 0.0188 |
| rio | Optical-only | 0.4217 | 0.2512 | 0.3148 | 0.1868 | 0.0208 |
| rio | Synchronous SAR + Optical Fusion | 0.5191 | 0.5330 | 0.5259 | 0.3568 | 0.0298 |
| saclay_w | SAR-only | 0.3239 | 0.5094 | 0.3960 | 0.2469 | 0.0123 |
| saclay_w | Optical-only | 0.0720 | 0.0998 | 0.0836 | 0.0436 | 0.0149 |
| saclay_w | Synchronous SAR + Optical Fusion | 0.1078 | 0.2529 | 0.1511 | 0.0817 | 0.0242 |
| valencia | SAR-only | 0.0000 | 0.0000 | undefined | 0.0000 | 0.0023 |
| valencia | Optical-only | 0.0080 | 0.1321 | 0.0150 | 0.0076 | 0.0734 |
| valencia | Synchronous SAR + Optical Fusion | 0.0113 | 0.2621 | 0.0217 | 0.0110 | 0.1022 |
| dubai | SAR-only | 0.4688 | 0.4574 | 0.4630 | 0.3012 | 0.0571 |
| dubai | Optical-only | 0.2328 | 0.0811 | 0.1203 | 0.0640 | 0.0295 |
| dubai | Synchronous SAR + Optical Fusion | 0.2694 | 0.1315 | 0.1768 | 0.0970 | 0.0393 |
| lasvegas | SAR-only | 0.6588 | 0.1969 | 0.3032 | 0.1787 | 0.0085 |
| lasvegas | Optical-only | 0.4986 | 0.5466 | 0.5215 | 0.3527 | 0.0457 |
| lasvegas | Synchronous SAR + Optical Fusion | 0.2873 | 0.8725 | 0.4322 | 0.2757 | 0.1799 |
| milano | SAR-only | 0.1777 | 0.3084 | 0.2255 | 0.1271 | 0.0114 |
| milano | Optical-only | 0.4167 | 0.0062 | 0.0122 | 0.0061 | 0.0001 |
| milano | Synchronous SAR + Optical Fusion | 0.6065 | 0.1166 | 0.1956 | 0.1084 | 0.0006 |
| chongqing | SAR-only | 0.4108 | 0.2703 | 0.3261 | 0.1948 | 0.0301 |
| chongqing | Optical-only | 0.9370 | 0.0472 | 0.0900 | 0.0471 | 0.0002 |
| chongqing | Synchronous SAR + Optical Fusion | 0.6838 | 0.2276 | 0.3415 | 0.2059 | 0.0082 |

## Unweighted mean ± sample standard deviation

| Model | Precision | Recall | F1 | IoU | FAR |
| --- | ---: | ---: | ---: | ---: | ---: |
| SAR-only | 0.3872 ± 0.1865 (n=10) | 0.2377 ± 0.1619 (n=10) | 0.2902 ± 0.1136 (n=9) | 0.1568 ± 0.0908 (n=10) | 0.0156 ± 0.0169 (n=10) |
| Optical-only | 0.3438 ± 0.3068 (n=10) | 0.2882 ± 0.2584 (n=10) | 0.2076 ± 0.2185 (n=10) | 0.1333 ± 0.1605 (n=10) | 0.0568 ± 0.0945 (n=10) |
| Synchronous SAR + Optical Fusion | 0.3502 ± 0.2258 (n=10) | 0.4120 ± 0.2639 (n=10) | 0.3080 ± 0.1835 (n=10) | 0.1952 ± 0.1358 (n=10) | 0.0511 ± 0.0546 (n=10) |

## Per-city F1 range

- SAR-only: highest recorded F1 is dubai (0.463003); lowest recorded F1 is brasilia (0.065106). Undefined F1 cities: valencia.
- Optical-only: highest recorded F1 is montpellier (0.649167); lowest recorded F1 is milano (0.012220). Undefined F1 cities: none.
- Synchronous SAR + Optical Fusion: highest recorded F1 is montpellier (0.627587); lowest recorded F1 is valencia (0.021698). Undefined F1 cities: none.

## Fusion F1 minus SAR-only F1

The difference is fusion F1 minus SAR-only F1. Cities are listed by absolute difference.

- montpellier: +0.387075
- dubai: -0.286236
- rio: +0.270958
- saclay_w: -0.244829
- brasilia: +0.226933
- lasvegas: +0.129064
- milano: -0.029918
- norcia: -0.021572
- chongqing: +0.015438

Undefined difference because one F1 value is undefined: valencia.

## Checks

- SAR-only: recorded best validation epoch 21 matches the maximum finite `validation_f1` in `training_history.csv`: True.
  Checkpoint selection text: highest validation F1 on full validation-city images
  Test aggregation text: pixel counts pooled across the official test cities, once, after training
  `training_history.csv` contains validation columns and no per-epoch test metrics.
  No warnings field is present in metrics.json.
  Undefined history cells: 6. Non-finite history values: 0.
  Undefined history cells: epoch 1 validation_precision, epoch 1 validation_f1, epoch 2 validation_precision, epoch 2 validation_f1, epoch 3 validation_precision, epoch 3 validation_f1.
- Optical-only: recorded best validation epoch 50 matches the maximum finite `validation_f1` in `training_history.csv`: True.
  Checkpoint selection text: highest validation F1 on full validation-city images
  Test aggregation text: pixel counts pooled across the official test cities, once, after training
  `training_history.csv` contains validation columns and no per-epoch test metrics.
  No warnings field is present in metrics.json.
  Undefined history cells: 8. Non-finite history values: 0.
  Undefined history cells: epoch 1 validation_precision, epoch 1 validation_f1, epoch 2 validation_precision, epoch 2 validation_f1, epoch 3 validation_precision, epoch 3 validation_f1, epoch 4 validation_f1, epoch 6 validation_f1.
- Synchronous SAR + Optical Fusion: recorded best validation epoch 46 matches the maximum finite `validation_f1` in `training_history.csv`: True.
  Checkpoint selection text: highest validation F1 on full validation-city images
  Test aggregation text: pixel counts pooled across the official test cities, once, after training
  `training_history.csv` contains validation columns and no per-epoch test metrics.
  Warnings field is empty.
  Undefined history cells: 6. Non-finite history values: 0.
  Undefined history cells: epoch 1 validation_precision, epoch 1 validation_f1, epoch 2 validation_precision, epoch 2 validation_f1, epoch 3 validation_f1, epoch 6 validation_f1.

SAR-only has an undefined test F1 for valencia. Optical-only and fusion have a recorded F1 for every test city. No NaN or Inf token appears in the three city tables.
