"""Scenario 2 — SAR-guided Cross-Attention Transformer training script.

STATUS: STUB — not ready to run.
The dataset loader (src/datasets/scenario2_dataset.py) does not exist yet.
It cannot be written until the dataset archive has been inspected per
docs/SCENARIO_2_DATASET_INSPECTION.md.

This file documents the intended structure and all known invariants so that
implementation can proceed quickly once the dataset is understood.

DO NOT modify any Scenario-1 files to make this script work.

Invariants (same as Scenario 1, will not change):
- Model: TransformerCrossAttention from src/models/transformer_cross_attention.py
- SAR → Query, Historical Optical → Key/Value (unchanged from Scenario 1)
- Input channels: 2 SAR (T1+T2) + N optical (T1 only) — N unknown until inspection
  If VH is available, SAR channels become 4 (T1-VV, T1-VH, T2-VV, T2-VH).
  Changing SAR channels requires adjusting TransformerCrossAttention(sar_channels=4).
  This is a Scenario-2 change only.
- S2 T2 must NEVER be loaded or passed to the model
- Loss: BCEWithLogitsLoss(pos_weight=computed_from_training_split)
- Optimizer: Adam
- Threshold: 0.5
- Checkpoint selection: highest validation F1
- last.pt saved unconditionally every epoch
- best.pt saved when validation F1 improves
- Outputs to: experiments/scenario_2/ and checkpoints/scenario_2/
- Seed: 42

Reference Scenario-1 hyperparameters (starting point for sweep):
- embed_dim: 128
- patch_stride: 4
- num_self_blocks: 2
- num_cross_blocks: 1
- num_heads: 4
- mlp_ratio: 4
These may be adjusted by the hyperparameter sweep (see
docs/SCENARIO_2_HYPERPARAMETER_SWEEP.md).

Unknown until dataset inspection:
- SAR channels per date (VV only? VV+VH?)
- Optical bands (N)
- Number of training/validation/test sites
- pos_weight value
- Whether 32×32 crop size is appropriate

TODO after dataset inspection:
1. Implement src/datasets/scenario2_dataset.py
2. Replace the TODO_DATASET placeholder below with the real dataset class
3. Compute pos_weight from training split
4. Set sar_channels based on available polarizations
5. Verify channel counts in run_verification()
6. Run --verify-only before any training
"""

# ── placeholder imports — uncomment and complete after dataset loader exists ──
# from src.datasets.scenario2_dataset import (
#     Scenario2Dataset,
#     TODO_TRAIN_SITES,
#     TODO_VAL_SITES,
#     TODO_TEST_SITES,
# )

raise NotImplementedError(
    "train_scenario2_cross_attention.py is a stub. "
    "Complete src/datasets/scenario2_dataset.py first. "
    "See docs/SCENARIO_2_DATASET_INSPECTION.md."
)
