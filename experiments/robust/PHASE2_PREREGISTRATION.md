# Phase 2 pre-registration: full-input vs modality-dropout training

Written 2026-10-21, before any Phase 2 run started. Changing anything below after results
exist must be reported as a deviation.

## Frozen configuration (shared by both arms)

- Code: `src/training/train_robust.py` sha256 `c7d812e3fd8dcf2c9bd8c7f4d3faeb1bee0c97fc256f83a76ee2194e15a6436c`;
  `src/datasets/oscd_dataset.py` `bc9b64bc…6fc43`; DS_UNet@8bfa4be `utils/networks.py` `a3e7216b…66830`,
  `utils/loss_functions.py` `6e71a2e8…4f204a7f`. Each run also stores its own snapshot and hashes.
- Network: Hafner DS-UNet (two U-Nets [64,128,256,512], late fusion by 1×1 conv), loaded from the DS_UNet clone.
- Data: OSCD, S1 VV + S2 13 bands per date. Protocol `holdout`: 10 train cities, 4 validation cities
  (aguasclaras, beihai, bercy, paris), 10 official test cities.
- Recipe: Jaccard-like loss, Adam lr 5e-5, weight decay 5e-4, batch 64, 32×32 importance crops,
  flip + rot90 augmentation, **2500 steps**, validation every 25 steps.
- Step budget chosen on **validation** score only (390: 0.354, 1000: 0.382, 2500: 0.405; mean of 3 seeds).
  An earlier informal choice of 1000 steps looked at test F1 and was discarded.
- Seeds 1–10 for both arms (paired).

## Arms

| arm | name | `--train-conditions` |
|---|---|---|
| A | `p2_full` | `full=1` |
| B | `p2_dropout` | `full=0.5,no_o2=0.25,no_opt=0.25` |

Probabilities for B were fixed a priori (half full input, the rest split equally between the two
missing-optical cases). They are not tuned.

## Selection rules (validation cities only)

- Checkpoint: step with the highest mean validation F1 (at each condition's best validation threshold)
  over the conditions the arm trains on: A uses `full`; B uses `full`, `no_o2`, `no_opt`.
- Threshold: per test condition, the value in {0, 0.01, …, 1} that maximises validation F1 for the selected checkpoint.
- Test cities are evaluated once per run, after training.

## Test conditions

`full`; `no_o2` (S2 t2 zero-filled); `o2_copy` (S2 t2 replaced by S2 t1, an inference-time imputation
that neither arm trains on); `no_opt` (all S2 zero-filled). Zero-filling simulates *missing*
imagery. It is not a model of cloud-contaminated imagery.

## Primary analysis

Per condition, paired difference B − A in test F1 at the validation-selected threshold over seeds 1–10:
mean, paired t-test and Wilcoxon signed-rank. Four conditions are tested, so Holm correction is applied
across them. F1 at 0.5, precision, recall, IoU and confusion counts are reported for every seed.

Claim rule: B is "more robust" under a condition only if the Holm-adjusted paired p < 0.05 and the mean
difference is positive. Any cost on `full` input is reported whether or not it is significant.
