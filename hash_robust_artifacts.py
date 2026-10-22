"""sha256 of every robustness-study artifact that is not committed to git (binary arrays, checkpoints, catalogue archive).

Writes experiments/robust/ARTIFACT_SHA256_<label>.txt in `sha256sum` format (relative paths), so `sha256sum -c` verifies it.
Refuses to overwrite an existing file with the same label.

Usage:
    .venv/Scripts/python.exe hash_robust_artifacts.py <label>
"""

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PATTERNS = ("experiments/robust/**/*.npy", "experiments/robust/**/*.npz", "checkpoints/robust/*.pt",
            "datasets/s2_cloud_mask_catalogue/masks.zip")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


if __name__ == "__main__":
    out = ROOT / "experiments" / "robust" / f"ARTIFACT_SHA256_{sys.argv[1]}.txt"
    if out.exists():
        raise SystemExit(f"{out.name} exists; refusing to overwrite")
    files = sorted({p for pat in PATTERNS for p in ROOT.glob(pat)})
    out.write_text("".join(f"{sha256(p)} *{p.relative_to(ROOT).as_posix()}\n" for p in files), encoding="utf-8")
    print(f"{len(files)} files -> {out.relative_to(ROOT).as_posix()}")
