"""Verify every sha256 in a pre-registration's hash table against the files on disk. Exits 1 on any mismatch.

Rows look like | `path` | `sha256` |. A path may carry a parenthesised note ("(B)"); a glob row
"dir/*.npy (N, concatenated, sorted)" is the sha256 of the sorted files' bytes concatenated. Paths are tried relative to
the project root, then experiments/robust/, then the parent of the project's grandparent (repos/...).

Usage:
    .venv/Scripts/python.exe verify_prereg_hashes.py experiments/robust/PHASE4B_PREREGISTRATION.md
"""

import glob
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASES = (ROOT, ROOT / "experiments" / "robust", ROOT.parents[1])


def digest(paths):
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


def main(prereg: str) -> int:
    rows = re.findall(r"^\| `([^`]+)` \| `([0-9a-f]{64})` \|", Path(prereg).read_text(encoding="utf-8"), re.M)
    bad = 0
    for raw, sha in rows:
        path = raw.split(" (")[0]
        found = None
        for base in BASES:
            files = sorted(glob.glob(str(base / path))) if "*" in path else ([base / path] if (base / path).exists() else [])
            if files:
                found = files
                break
        ok = found is not None and digest(found) == sha
        bad += not ok
        print(("ok   " if ok else "FAIL ") + raw)
    print(f"{len(rows)} hashes, {bad} failed")
    return 1 if bad or not rows else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
