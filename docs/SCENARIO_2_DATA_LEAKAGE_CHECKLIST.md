# Scenario 2 — Data Leakage Checklist

Updated after dataset inspection.
Verify every item before running any Scenario 2 training.

---

## A. Scene/site separation

- [ ] The 56/12/12 site split is defined using seed 42 before any training data is examined.
- [ ] Site list is written to `experiments/scenario_2/configs/split_definition.json` and not modified after training begins.
- [ ] No site ID appears in more than one of train / validation / test.
- [ ] Sites with identical or overlapping geographic tile IDs are checked and assigned to the same split.

---

## B. Temporal leakage

- [ ] T1 is defined as the earliest unmasked timestep per site, T2 as the latest.
- [ ] T1 year/month is verified to be earlier than T2 year/month for every site (can be checked from metadata.json).
- [ ] The change label is derived from (buildings_T2 ≥ 0.5) XOR (buildings_T1 ≥ 0.5) using the selected T1 and T2.
- [ ] The derivation threshold (0.5) is applied consistently at load time, not at evaluation time.
- [ ] No buildings label from after the change period is used as SAR context.

---

## C. Contemporary optical T2 not entering asynchronous input

- [ ] The async dataset loader only opens the S2 file at T1 (`s2_{site}_{T1_year}_{T1_month:02d}.tif`).
- [ ] The S2 file at T2 is never opened, read, or included in the input tensor for async models.
- [ ] An AST-level code check (as used in Scenario 1) confirms no T2 optical path appears in the async preprocessing function before training.
- [ ] The channel layout is verified: input shape is (B, 14, H, W) = 4 SAR + 10 optical, not 24.

---

## D. Test set not used for hyperparameter selection

- [ ] The test split is defined and locked before any training begins.
- [ ] No test-site label is examined, plotted, or evaluated during the hyperparameter sweep.
- [ ] The sweep (S2-ref-* through S2-pw-high) is evaluated exclusively on the 12 validation sites.
- [ ] The configuration is locked after the sweep. Test evaluation runs exactly once.

---

## E. Normalisation statistics

- [ ] S1 is already normalised to [0,1] in the dataset files. No additional normalisation is applied.
- [ ] S2 is already normalised to [0,1]. No division by 10000. This is verified at inspection (range confirmed [0,1]).
- [ ] No per-channel mean/std normalisation is applied (avoids needing to compute from train split).
- [ ] If per-channel normalisation is added in the future, statistics must be computed from the training split only.

---

## F. pos_weight calculation

- [ ] The no-change / change pixel ratio is computed from the training split only (56 sites, T1→T2 change masks).
- [ ] The ratio is recorded in `experiments/scenario_2/configs/class_balance.json` before training.
- [ ] The same pos_weight is used for all models in the sweep.
- [ ] The pos_weight is not recomputed after examining validation or test results.

---

## G. NaN handling

- [ ] S1 NaN pixels are filled with 0.0 in the dataset loader before tensor construction.
- [ ] The bad_data.json timestep indices are used to avoid loading known-bad files as T1 or T2.
- [ ] NaN-filled pixels do not receive gradient from the loss (verify that NaN→0 substitution does not introduce phantom change labels in the buildings mask).

---

## H. Model selection

- [ ] best.pt is selected by validation F1. The test set is not involved.
- [ ] After configuration is locked and best.pt is selected, test evaluation runs once.
- [ ] Results are reported without further modification.
