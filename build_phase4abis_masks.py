"""Build the Step 4A-bis detector masks from the Phase 3 oracle masks (no model, no label is read).

Writes experiments/robust/phase4abis/{masks/<city>_<level>_<variant>.npy, mask_manifest.json}; refuses to overwrite.

Usage:
    .venv/Scripts/python.exe build_phase4abis_masks.py
"""

import hashlib
import json

import numpy as np

from src.phase3.corruption import PHASE3
from src.phase3.mask_errors import VARIANTS, error_stats, make_detector_mask, variant_seed

OUT = PHASE3.parent / "phase4abis"

if __name__ == "__main__":
    if (OUT / "masks").exists():
        raise SystemExit("phase4abis masks exist; refusing to overwrite")
    (OUT / "masks").mkdir(parents=True)
    manifest = []
    for e in json.loads((PHASE3 / "mask_manifest.json").read_text(encoding="utf-8")):
        m = np.load(PHASE3 / "masks" / e["file"])
        for v in VARIANTS:
            p = make_detector_mask(m, e["city"], e["level"], v)
            path = OUT / "masks" / f"{e['city']}_{e['level']}_{v}.npy"
            np.save(path, p)
            manifest.append({"split": e["split"], "city": e["city"], "level": e["level"], "variant": v,
                             "seed": variant_seed(e["city"], e["level"], v) if v.startswith("miss") else None,
                             "oracle_sha256": e["sha256"], **error_stats(m, p), "file": path.name,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (OUT / "mask_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(len(manifest), "detector masks written")
