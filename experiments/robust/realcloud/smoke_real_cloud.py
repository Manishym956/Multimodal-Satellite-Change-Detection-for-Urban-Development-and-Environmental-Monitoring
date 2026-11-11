"""Step 11 smoke test. Prints pass/fail only, no metric; exits 1 on any failure.

1-3  eval_real_cloud.py on seed 1 of A, B and SAR-only into a temporary folder (deleted afterwards; counts not read
     beyond their keys): exits cleanly, every anchor passes, every expected city x condition array exists.
4    eval_real_cloud.py refuses to overwrite.
5-7  analyze_real_cloud.py on synthetic random counts with the same layout: runs; a contrast between two identical
     conditions is exactly zero; refuses to overwrite.

Usage (repository root):  .venv/Scripts/python.exe experiments/robust/realcloud/smoke_real_cloud.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_real_cloud import CONDS, SETS  # noqa: E402

ARMS = ("p2_full", "p2_dropout", "p6_unet_sar")
checks = []


def check(name: str, ok: bool) -> None:
    checks.append(bool(ok))
    print(("ok   " if ok else "FAIL ") + name)


def run(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)


with tempfile.TemporaryDirectory() as tmp:
    T = Path(tmp)
    ev = run(HERE / "eval_real_cloud.py", "--seeds", "1-1", "--out", T / "runs")
    check("evaluation of seed 1 exits cleanly", ev.returncode == 0 and "REAL-CLOUD EVAL DONE" in ev.stdout)
    if ev.returncode:
        print((ev.stderr.strip().splitlines() or ["?"])[-1])
    metas = [json.loads((T / "runs" / f"{a}_seed1.json").read_text(encoding="utf-8")) for a in ARMS if (T / "runs" / f"{a}_seed1.json").exists()]
    check("every anchor passes for A, B and SAR-only", len(metas) == 3 and all(all(m["anchors"].values()) for m in metas))
    keys_ok = True
    for a in ARMS:
        f = T / "runs" / f"{a}_seed1.npz"
        conds = ("oscd", "real") if a == "p6_unet_sar" else CONDS
        keys_ok &= f.exists() and set(np.load(f).files) == {f"{c}__{k}" for c in SETS["all"] for k in conds}
    check("every city x condition array exists", keys_ok)
    again = run(HERE / "eval_real_cloud.py", "--seeds", "1-1", "--out", T / "runs")
    check("evaluation refuses to overwrite", again.returncode != 0 and "refusing to overwrite" in again.stdout + again.stderr)

with tempfile.TemporaryDirectory() as tmp:
    T = Path(tmp)
    (T / "runs").mkdir()
    rng = np.random.default_rng(0)
    for a in ARMS:
        for s in range(1, 11):
            arrs = {}
            for c in SETS["all"]:
                for k in (("oscd", "real") if a == "p6_unet_sar" else CONDS):
                    tp, fp = np.sort(rng.integers(0, 5000, 101))[::-1], np.sort(rng.integers(0, 50000, 101))[::-1]
                    arrs[f"{c}__{k}"] = np.stack([tp, fp, 6000 - tp, 60000 - fp], 1)
            if a == "p2_dropout":  # identical conditions: their contrast must be exactly zero
                for c in SETS["all"]:
                    arrs[f"{c}__real_bal"] = arrs[f"{c}__real"]
            np.savez(T / "runs" / f"{a}_seed{s}.npz", **arrs)
            (T / "runs" / f"{a}_seed{s}.json").write_text(json.dumps({"anchors": {"x": True}, "tau_clean": 50}), encoding="utf-8")
    an = run(HERE / "analyze_real_cloud.py", "--runs", T / "runs", "--out", T / "out")
    check("analysis runs on synthetic counts", an.returncode == 0)
    if an.returncode:
        print((an.stderr.strip().splitlines() or ["?"])[-1])
    sm = json.loads((T / "out" / "summary.json").read_text(encoding="utf-8")) if an.returncode == 0 else {}
    check("contrast of identical conditions is zero", sm and sm["secondary"]["B real_bal - B real"]["mean_diff"] == 0.0)
    again = run(HERE / "analyze_real_cloud.py", "--runs", T / "runs", "--out", T / "out")
    check("analysis refuses to overwrite", again.returncode != 0)

print(f"n_failed {checks.count(False)}")
sys.exit(1 if False in checks else 0)
