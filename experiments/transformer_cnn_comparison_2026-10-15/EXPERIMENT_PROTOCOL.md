# Frozen comparison protocol

## Scope and decision

Use existing, validated Transformer tuning results and immutable CNN checkpoints. Do not train new models in this comparison. Transformer runs already used a two-candidate seed-42 optimizer search per model and seeds 43/44 for the selected configuration. A matched CNN search is deferred: the synchronous dual-stream CNN baseline required 2,993.53 seconds for one run, and the training scripts automatically evaluate the test set at run end and hard-code their baseline output paths. A scientifically safe tuning fork would need a no-test model-selection path, checkpoint isolation, and parity across the CNNs; estimated four-run budget for the synchronous model alone is about 3.3 hours at baseline runtime, before repeated architectures or common-loss controls. CNNs are therefore explicitly fixed benchmarks, not tuned comparators. This is a resource/protocol decision, not evidence that CNNs could not improve.

## Data and splits

- OSCD: existing 10 train / 4 validation / 10 test city split, seed 42. Preserve canonical dataset loading and first-listed evaluation orbit. The CNN benchmark models use the same OSCD test cities. Clean asynchronous and Transformer models receive 2 SAR + 13 historical optical channels; synchronous CNN receives 2 SAR + 26 optical channels (both dates). The latter is a different information condition.
- Scenario 2: preserve split hash `5d0841adf0c651166cb18ecea310c6ef8cce69172065fc2d699a29083c272e27`; 20/4/4 geographic sites. Report only existing outputs; do not expand samples. Labels are a building-probability XOR proxy.
- Keep all historic test disclosures explicit. Any new calibration uses validation labels only. CNN validation and test inference each generate one probability map per city with `model.eval()` and `torch.inference_mode()`. Thresholds are calculated from these maps after inference. For the final test view, only .5 and the already-frozen validation threshold are tabulated; no test result changes the frozen threshold or model selection.

## Models and losses

Use original configurations/checkpoints unchanged for historical baseline rows. OSCD Transformers train with BCE `pos_weight=41`; OSCD CNN baselines train with unweighted BCE. Scenario 2 CNNs and cross-attention Transformer train with BCE `pos_weight=94`. Since loss weighting differs by OSCD family, the table is descriptive; do not attribute score gaps solely to attention. No common-loss retraining is included. Report the model size and exact input condition where available.

## Selection and evaluation

- Reuse selected Transformer checkpoints from `experiments/transformer_tuning_2026-10-15/`; valid seeds are 42, 43, 44. The configuration selection criterion was maximum pooled validation F1 at .5 among two registered settings. Per-run threshold was selected by maximum pooled validation F1 from the saved validation sweep.
- For OSCD clean asynchronous and synchronous dual-stream CNN baselines, calculate validation pooled F1 across a fixed threshold grid: .05–.95 by .05 plus .30–.70 by .0125. Select the maximum F1; ties select the higher threshold. Freeze it before test count calculation. This is an operating-point comparison only; it does not retune weights.
- View A compares all available checkpoint results at threshold .5. View B compares validation-thresholded Transformers and the two main OSCD CNN comparators. Other CNNs without validation calibration are `N/A` in View B.
- Metrics are pooled pixel-level precision, recall, F1, IoU, FAR = FP/(FP+TN), TP/FP/TN/FN. Report evaluated pixels where available. Pixel-level rates are not independent-city uncertainty estimates.
- Scenario 2 threshold view is reported from previously saved validation-calibration artifacts only when the valid-pixel policy and sample definition match. Do not mix it into the all-pixel primary table without a protocol label.

## Run registry and stopping rule

No new training configurations are planned. Preserve all 17 Transformer tuning run records, including five failed/invalidated attempts. Reuse the existing successful seed repeats and existing CNN seed repeats only where their metric protocol matches. Stop if a checkpoint is missing, split or checkpoint hash differs, a loader changes samples, or validation/test alignment fails; report `N/A` and the blocker rather than trying a different split.
