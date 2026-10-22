"""Step 4B: a fixed bank of cloud-shaped 32x32 training masks (no model, no OSCD data is read).

Source: Sentinel-2 Cloud Mask Catalogue masks, EXCLUDING every scene used by a Phase 3 mask (validation and test cities,
current and superseded builds), so no evaluation cloud shape is seen in training. Each draw (seed BANK_SEED): a uniform eligible scene, CLOUD channel upsampled x2
(20 m -> 10 m), a uniform 32x32 window; kept if its cloud fraction is in [0.05, 0.95] (partial masks only).

Writes experiments/robust/phase4b/{spatial_mask_bank.npy (N, 32, 32) bool, spatial_mask_bank.json}; refuses to overwrite.

Usage:
    .venv/Scripts/python.exe build_spatial_mask_bank.py
"""

import hashlib
import json

import numpy as np

from src.phase3.corruption import PHASE3, catalogue_mask_files, cloud_channel

OUT = PHASE3.parent / "phase4b"
BANK_SEED = 20261022
N = 20000
SIZE = 32
FRACTION = (0.05, 0.95)

if __name__ == "__main__":
    if (OUT / "spatial_mask_bank.npy").exists():
        raise SystemExit("bank exists; refusing to overwrite")
    OUT.mkdir(parents=True, exist_ok=True)
    # every scene behind a Phase 3 mask, validation or test, including the superseded ±0.05 build (never evaluated)
    used = {e["scene"] for m in (PHASE3 / "mask_manifest.json", PHASE3 / "masks_superseded_tol005" / "mask_manifest.json")
            for e in json.loads(m.read_text(encoding="utf-8"))}
    files = [f for f in catalogue_mask_files() if f.stem not in used]
    rng = np.random.default_rng(BANK_SEED)
    bank, sources, draws = [], [], 0
    while len(bank) < N:
        draws += 1
        f = files[rng.integers(len(files))]
        cloud = cloud_channel(f).repeat(2, 0).repeat(2, 1)
        y, x = rng.integers(0, cloud.shape[0] - SIZE + 1), rng.integers(0, cloud.shape[1] - SIZE + 1)
        w = cloud[y:y + SIZE, x:x + SIZE]
        if FRACTION[0] <= w.mean() <= FRACTION[1]:
            bank.append(w.copy())
            sources.append(f.stem)
    bank = np.stack(bank)
    np.save(OUT / "spatial_mask_bank.npy", bank)
    frac = bank.mean((1, 2))
    doc = {"n": N, "size": SIZE, "fraction_range": FRACTION, "seed": BANK_SEED, "draws": draws,
           "eligible_scenes": len(files), "excluded_eval_scenes": sorted(used), "scenes_used": len(set(sources)),
           "mean_fraction": float(bank.mean()),
           "fraction_quantiles": {str(q): float(np.quantile(frac, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
           "fraction_histogram_0.1": np.histogram(frac, bins=np.linspace(0, 1, 11))[0].tolist(),
           "max_scene_share": max(sources.count(s) for s in set(sources)) / N,
           "sha256": hashlib.sha256((OUT / "spatial_mask_bank.npy").read_bytes()).hexdigest()}
    assert not (set(sources) & used)
    (OUT / "spatial_mask_bank.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print({k: v for k, v in doc.items() if k != "excluded_eval_scenes"})
