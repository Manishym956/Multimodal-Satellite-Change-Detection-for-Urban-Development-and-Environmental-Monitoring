"""Step 10 smoke test. Uses only already-reported runs (Step 9 E and Step 6 SAR-only, seeds 1-10) relabelled as seeds
11-20 in a temporary folder that is deleted afterwards; prints pass/fail only, no metric. Exits 1 on any failure.

1  eval_step10_cities.py runs on the stand-in checkpoints and its anchors pass (city counts sum to stored test counts)
2  eval_step10_cities.py refuses to overwrite
3  analyze_step10.py runs end to end on the stand-in data
4  its primary contrast equals the same contrast computed directly from Step 9's summary
5  its all-city pooled delta from per-city counts equals the primary mean difference
6  analyze_step10.py refuses to overwrite

Usage (repository root):  .venv/Scripts/python.exe experiments/robust/step10/smoke_step10.py
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
R = ROOT / "experiments" / "robust"
STAND_IN = {"p10_fusion_dropout": "p9_fusion_dropout", "p10_unet_sar": "p6_unet_sar"}
checks = []


def check(name: str, ok: bool) -> None:
    checks.append(bool(ok))
    print(("ok   " if ok else "FAIL ") + name)


def run(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)


with tempfile.TemporaryDirectory() as tmp:
    T = Path(tmp)
    for new, old in STAND_IN.items():
        for s in range(1, 11):
            m = json.loads((R / f"{old}_seed{s}" / "metrics.json").read_text(encoding="utf-8"))
            m["args"]["seed"] = s + 10
            (T / "runs" / f"{new}_seed{s + 10}").mkdir(parents=True)
            (T / "runs" / f"{new}_seed{s + 10}" / "metrics.json").write_text(json.dumps(m), encoding="utf-8")
            (T / "ckpts").mkdir(exist_ok=True)
            shutil.copy(ROOT / "checkpoints" / "robust" / f"{old}_seed{s}.pt", T / "ckpts" / f"{new}_seed{s + 10}.pt")
    ev = [run(HERE / "eval_step10_cities.py", "--arm", a, "--seeds", "11-20", "--runs", T / "runs", "--ckpts", T / "ckpts",
              "--out", T / "cities") for a in STAND_IN]
    check("city evaluation runs and its anchors pass", all(e.returncode == 0 and "CITY EVAL DONE" in e.stdout for e in ev))
    again = run(HERE / "eval_step10_cities.py", "--arm", "p10_unet_sar", "--seeds", "11-11", "--runs", T / "runs",
                "--ckpts", T / "ckpts", "--out", T / "cities")
    check("city evaluation refuses to overwrite", again.returncode != 0 and "refusing to overwrite" in again.stdout + again.stderr)
    an = run(HERE / "analyze_step10.py", "--runs", T / "runs", "--step9-runs", R, "--cities", T / "cities", "--out", T / "out")
    check("analysis runs on stand-in data", an.returncode == 0)
    if an.returncode:  # the traceback names code, not metrics
        print(an.stderr.strip().splitlines()[-1])
    sm = json.loads((T / "out" / "summary.json").read_text(encoding="utf-8"))
    s9 = json.loads((R / "step9" / "summary.json").read_text(encoding="utf-8"))["arms"]
    direct = np.mean(np.array(s9["p9_fusion_dropout"]["at_selected"]["no_opt"]["f1"]["values"])
                     - np.array(s9["p6_unet_sar"]["at_selected"]["full"]["f1"]["values"]))
    prim = sm["primary"]["E@no_opt - SAR-only"]["mean_diff"]
    check("primary contrast reproduces the Step 9 values", abs(prim - direct) < 1e-12)
    check("pooled delta from per-city counts equals the primary mean difference", abs(sm["city_level"]["delta"] - prim) < 1e-12)
    again = run(HERE / "analyze_step10.py", "--runs", T / "runs", "--cities", T / "cities", "--out", T / "out")
    check("analysis refuses to overwrite", again.returncode != 0)

print(f"n_failed {checks.count(False)}")
sys.exit(1 if False in checks else 0)
