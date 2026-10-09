"""Scenario 2 — Clean Asynchronous Dual-Stream CNN training script.

STATUS: STUB — not ready to run.
The dataset loader (src/datasets/scenario2_dataset.py) does not exist yet.
It cannot be written until the dataset archive has been inspected per
docs/SCENARIO_2_DATASET_INSPECTION.md.

This file documents the intended structure and all known invariants so that
implementation can proceed quickly once the dataset is understood.

DO NOT modify any Scenario-1 files to make this script work.

Invariants (same as Scenario 1, will not change):
- Model: DualStreamUNetClean from src/models/dual_stream_unet_clean.py
- Input channels: 2 SAR (T1+T2) + N optical (T1 only) — N unknown until inspection
- S2 T2 must NEVER be loaded or passed to the model
- Loss: BCEWithLogitsLoss(pos_weight=computed_from_training_split)
- Optimizer: Adam
- Threshold: 0.5
- Checkpoint selection: highest validation F1
- last.pt saved unconditionally every epoch
- best.pt saved when validation F1 improves
- Outputs to: experiments/scenario_2/ and checkpoints/scenario_2/
- Seed: 42

Unknown until dataset inspection:
- SAR channels per date (VV only? VV+VH?)
- Optical bands (N)
- Number of training/validation/test sites
- pos_weight value
- Whether 32×32 crop size is appropriate or needs adjustment

TODO after dataset inspection:
1. Implement src/datasets/scenario2_dataset.py
2. Replace the TODO_DATASET placeholder below with the real dataset class
3. Compute pos_weight from training split
4. Verify channel counts in run_verification()
5. Run --verify-only before any training
"""

# ── placeholder imports — uncomment and complete after dataset loader exists ──
# from src.datasets.scenario2_dataset import (
#     Scenario2Dataset,
#     TODO_TRAIN_CITIES,
#     TODO_VAL_CITIES,
#     TODO_TEST_CITIES,
# )

raise NotImplementedError(
    "train_scenario2_dual_stream.py is a stub. "
    "Complete src/datasets/scenario2_dataset.py first. "
    "See docs/SCENARIO_2_DATASET_INSPECTION.md."
)
