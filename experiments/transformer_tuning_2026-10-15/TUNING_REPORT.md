# Controlled Transformer Tuning — Results

**Run date:** 2026-10-15  
**Decision:** preliminary, configuration-level tuning evidence only. The models were tuned on the existing OSCD and Scenario 2 samples. Scenario 2 provenance remains unresolved, and this work does not answer RQ2 temporal-gap sensitivity.

## Scope and safeguards

Tuned OSCD self-attention, OSCD cross-attention, and Scenario 2 cross-attention. There is no Scenario 2 self-attention baseline/checkpoint, so that experiment was not fabricated. The GPU was an NVIDIA RTX 4060 Laptop GPU (8 GB); Python 3.11.0, PyTorch 2.5.1+cu124, CUDA available. Each valid run used 50 epochs maximum, batch 64, 32-pixel crops, 64 crops per city/site, Adam, BCE with `pos_weight=41` (OSCD) or `94` (Scenario 2), and validation pooled F1 at threshold .5 for checkpoint/config selection. The selected validation thresholds were then chosen by maximizing validation pooled F1 on a .05-to-.80 sweep with .0125/.025 intermediate grid points near peaks. No test scores selected a model, checkpoint, or threshold. Test results were calculated after the configuration and per-run validation threshold were frozen.

The selected seed-42 settings were OSCD self: LR `1e-4`, WD `1e-4`; OSCD cross: LR `1e-4`, WD `1e-4`; Scenario 2 cross: LR `3e-4`, WD `0`. Seeds 43 and 44 repeated each selected setting. These runs change optimizer hyperparameters only; no architecture redesign, loss change, gate, or Scenario 2 self-attention experiment was performed. Baselines and tuned models were evaluated with the same pooled metrics code. Scenario 2 evaluation has 954,528 total pixels and 825,268 SAR-valid pixels; primary reported confusion metrics use the existing all-target-pixel policy and valid-pixel counts are also retained in the CSV.

## Verified architecture and baseline settings

The OSCD inputs have 15 channels: two SAR VV channels followed by 13 historical Sentinel-2 channels. Scenario 2 uses 14 channels: four SAR channels followed by 10 configured optical bands. Both implementations use 128-dimensional patch embeddings with stride 4, four heads, two pre-LayerNorm self-attention blocks per modality, MLP ratio 4, dropout 0 in the recorded baseline and tuning configs, and a convolutional upsampling decoder returning a one-channel full-resolution logit mask. The cross model adds one pre-LayerNorm cross-attention block with SAR as query and optical as key/value, residual updates and an MLP refinement; it decodes the updated SAR stream. The self model concatenates the two encoded streams before decoding. No explicit positional encoding is implemented in these model classes. Decoder blocks include BatchNorm; this is why the initial tuning driver's validation-mode bug invalidated affected runs. Parameter counts are self 1,358,081; OSCD cross 1,156,993; Scenario 2 cross 1,154,945. Baseline optimizer settings were Adam, LR `1e-4`, WD `0`, BCE with `pos_weight=41` (OSCD) or `94` (Scenario 2), batch 64 and 50 epochs maximum. Original validation-best epochs were 37 (OSCD self), 33 (OSCD cross), and 23 (Scenario 2 cross). The training configs preserve the exact per-run code hashes and split identifiers.

## Locked test comparison

Precision, recall, F1, IoU are pooled. FP and FN are pixel counts; FAR is FP/(FP+TN), not raw FP. “Tuned” below is seed 42. `Δ F1` is against the original checkpoint at threshold .5.

| Dataset / model | Condition | Threshold | Precision | Recall | F1 | IoU | FP | FN | FAR | Δ F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OSCD self | Original | .5000 | .0974 | .4734 | .1615 | .0879 | 697,994 | 83,770 | .2391 | — |
| OSCD self | Original, val threshold | .6125 | .0965 | .4064 | .1559 | .0846 | 605,324 | 94,436 | .2074 | -.0056 |
| OSCD self | Tuned seed 42 | .5000 | .1428 | .6750 | .2357 | .1336 | 644,509 | 51,702 | .2208 | +.0742 (+45.9%) |
| OSCD self | Tuned seed 42, val threshold | .4375 | .1333 | .7271 | .2253 | .1269 | 752,086 | 43,415 | .2577 | +.0637 (+39.5%) |
| OSCD cross | Original | .5000 | .2615 | .5593 | .3564 | .2168 | 251,220 | 70,112 | .0861 | — |
| OSCD cross | Original, val threshold | .5625 | .3213 | .4807 | .3851 | .2385 | 161,557 | 82,610 | .0553 | +.0287 |
| OSCD cross | Tuned seed 42 | .5000 | .1578 | .5755 | .2477 | .1414 | 488,537 | 67,522 | .1674 | -.1087 (-30.5%) |
| OSCD cross | Tuned seed 42, val threshold | .5250 | .1691 | .5563 | .2594 | .1490 | 434,796 | 70,586 | .1490 | -.0970 (-27.2%) |
| Scenario 2 cross | Original | .5000 | .0214 | .5705 | .0412 | .0210 | 179,292 | 2,946 | .1892 | — |
| Scenario 2 cross | Original, val threshold | .7250 | .0260 | .3925 | .0487 | .0250 | 100,964 | 4,167 | .1065 | +.0075 (+18.3%) |
| Scenario 2 cross | Tuned seed 42 | .5000 | .0494 | .3043 | .0850 | .0444 | 40,158 | 4,772 | .0424 | +.0438 (+106.4%) |
| Scenario 2 cross | Tuned seed 42, val threshold | .6000 | .0562 | .2371 | .0908 | .0476 | 27,331 | 5,233 | .0288 | +.0496 (+120.5%) |

Threshold selection alone improved test F1 for OSCD cross and Scenario 2 cross, but hurt OSCD self. Training improved seed-42 test F1 at .5 for OSCD self and Scenario 2 cross and regressed OSCD cross. These are single selected-seed outcomes, not evidence of a consistent architecture-wide effect.

## Repeatability

| Dataset / model | Test F1 @ .5, seeds 42/43/44 | Mean ± sample SD | Per-seed validation-selected threshold F1 | Mean ± sample SD |
|---|---|---:|---|---:|
| OSCD self | .2357 / .2119 / .1870 | .2115 ± .0244 | .2253 / .2103 / .2038 | .2131 ± .0110 |
| OSCD cross | .2477 / .2852 / .1593 | .2307 ± .0647 | .2594 / .3267 / .2547 | .2803 ± .0403 |
| Scenario 2 cross | .0850 / .0656 / .0688 | .0731 ± .0104 | .0908 / .0623 / .0715 | .0749 ± .0146 |

Repeat-seed results vary, especially OSCD cross. The small n does not support significance claims. Validation-selected thresholds varied by seed: OSCD self `.4375/.4875/.575`; OSCD cross `.525/.575/.725`; Scenario 2 cross `.60/.3875/.65`.

## Baseline reproduction and run issues

The original checkpoint test reruns at .5 reproduce OSCD self and Scenario 2 cross recorded counts. OSCD cross differs slightly from its saved metrics: rerun TP/TN/FP/FN `88,965 / 2,667,639 / 251,220 / 70,112`; saved counts `88,959 / 2,667,663 / 251,196 / 70,118`. F1 differs by about `0.0000026`. The checkpoint, metrics, and per-city hashes match the existing artifact-integrity inventory; this small result discrepancy is retained rather than hidden.

Five trial directories were invalidated and excluded from model selection: one failed after epoch 1 due to a driver formatting exception; three completed or partially completed with validation run in training mode (BatchNorm state defect); one was interrupted at epoch 22 after that defect was found. Those folders and failure records are preserved. Six valid seed-42 candidates and six seed-43/44 repeats completed. No test split was used for candidate selection.

The original Scenario 2 cross baseline has 1,154,945 parameters; OSCD cross 1,156,993; OSCD self 1,358,081. Selected configurations preserve these architectures. Peak VRAM was approximately 803 MB (self), 482.5 MB (OSCD cross), and 294.6 MB (Scenario 2 cross). Valid tuning runs took about 307–352 seconds per 50-epoch run. No optional gate or Tversky experiment was justified after the mixed results and compute already used.

## Limitations and decision

The Scenario 2 archive remains unverified and the meanings of `s2`, `masked`, and bad-S2 indices remain unresolved. The tuning used the previously audited fixed sample set and did not change eligibility. It must not be presented as evidence about temporal-gap sensitivity or environmental-change ground truth. The RQ2 temporal-gap design remains blocked by provenance and the all-gap fixed-site intersection of only 7 sites (5 train, 1 validation, 1 test); changing sites across gaps would confound gap with site composition.

Before temporal-gap training, obtain and verify the exact upstream archive against published MD5 `5d34fc4537c7974c31824baaaea050f5` (or trusted per-file manifest); obtain authoritative definitions of `s2`, `masked`, and bad-S2 indices including index base/order and the 85 conflicting records' dispositions; rerun eligibility audit after resolving those flags; and predeclare a feasible gap set and site strategy with per-split counts. Archive identity alone does not explain record-level flag decisions.

## Artifacts

- `FINAL_TUNING_RESULTS.csv`: all baseline/tuned threshold conditions, metrics, counts, run paths, runtime, VRAM, and changes.
- `REPEATABILITY_SUMMARY.csv`: per-seed means and sample SDs.
- `TRAINING_RUNS.csv`: all valid, failed, and invalidated candidate runs.
- `threshold_validation_sweep.csv`, `selected_baseline_thresholds.json`, `baseline_test_thresholds.csv`: original-model threshold stage.
- `selected_tuning_configs.json`, `final_test_comparison.csv`: frozen selection and final test results.
- `test_f1_comparison.png`: threshold decomposition and seed variability.
- `runs/<run-id>/config.json`, `training_history.csv`, `result.json`, `best.pt`, `last.pt`: per-run configuration, logs, metrics, and checkpoints where completed.
- Locked split: SHA-256 `5d0841adf0c651166cb18ecea310c6ef8cce69172065fc2d699a29083c272e27`.
