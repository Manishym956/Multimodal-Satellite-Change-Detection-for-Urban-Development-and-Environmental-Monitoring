"""Recover the catalogue scenes behind the Step 4B mask bank by replaying build_spatial_mask_bank.py exactly.

build_spatial_mask_bank.py recorded how many scenes it used (153) but not which. This replays the same RNG stream with the
same code path, rebuilds the bank in memory, and accepts the recovered scene list only if the rebuilt bank's sha256
equals the recorded one (phase4b/spatial_mask_bank.json). Writes phase4b/spatial_mask_bank_sources.json; refuses to overwrite.
Nothing in phase4b/ is modified.

Usage:
    .venv/Scripts/python.exe replay_spatial_mask_bank_sources.py
"""

import hashlib
import io
import json

import numpy as np

from build_spatial_mask_bank import BANK_SEED, FRACTION, N, OUT, SIZE
from src.phase3.corruption import PHASE3, catalogue_mask_files, cloud_channel

if __name__ == "__main__":
    dest = OUT / "spatial_mask_bank_sources.json"
    if dest.exists():
        raise SystemExit("sources file exists; refusing to overwrite")
    recorded = json.loads((OUT / "spatial_mask_bank.json").read_text(encoding="utf-8"))
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
    buf = io.BytesIO()
    np.save(buf, np.stack(bank))
    sha = hashlib.sha256(buf.getvalue()).hexdigest()
    ok = sha == recorded["sha256"] and draws == recorded["draws"] and len(set(sources)) == recorded["scenes_used"]
    print(f"replayed sha256 {sha}, draws {draws}, scenes {len(set(sources))}; matches recorded bank: {ok}")
    if not ok:
        raise SystemExit("replay does not reproduce the recorded bank; no sources file written")
    dest.write_text(json.dumps({"bank_sha256": sha, "draws": draws, "scenes": sorted(set(sources)),
                                "per_scene_count": {s: sources.count(s) for s in sorted(set(sources))}}, indent=2), encoding="utf-8")
    print("wrote", dest)
