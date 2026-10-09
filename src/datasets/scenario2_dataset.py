"""Scenario 2 dataset loader.

STATUS: STUB — not ready to use.
This file cannot be implemented until the dataset archive has been inspected.
See docs/SCENARIO_2_DATASET_INSPECTION.md.

When implementation begins, this module must:
1. Load SAR T1 and SAR T2 (VV only, or VV+VH if available — check inspection results).
2. Load Sentinel-2 T1 (historical optical only). Do NOT load S2 T2.
3. Load the binary change mask.
4. Return a dict with keys: 'image', 'label', 'site', matching the
   structure expected by the training scripts.
5. Apply the same preprocessing normalisation as determined from the dataset
   inspection (band scaling, SAR range, etc.).

Imports from src/datasets/oscd_dataset.py are allowed if the normalisation
steps are identical. Do not modify oscd_dataset.py itself.

TODO after dataset inspection:
- Document DATASET_ROOT path convention
- Document SAR_CHANNELS (2 or 4)
- Document OPTICAL_CHANNELS (number of Sentinel-2 bands available)
- Document LABEL_REMAP (whether raw label values need +/- 1 like OSCD)
- Implement load_sites(split) equivalent
- Implement preprocess_site() equivalent
- Implement Scenario2Dataset(split, mode) class
"""

raise NotImplementedError(
    "scenario2_dataset.py is a stub. "
    "Implement after completing docs/SCENARIO_2_DATASET_INSPECTION.md."
)
