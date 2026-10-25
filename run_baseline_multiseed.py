"""Generic multi-seed runner for any OSCD training script that writes to a fixed
experiments/<name>/ and checkpoints/<name>/ directory.

Before the first run, an existing seed-42 base dir is archived as <name>_seed42
so it is never overwritten. Each seed's artifacts are copied to <name>_seed<N>,
then <name>_multiseed_summary.json is rebuilt from every <name>_seed*/metrics.json.

Usage:
    .venv/Scripts/python.exe run_baseline_multiseed.py --script train_sar_baseline --name sar_baseline --seeds 43 44
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENTS = PROJECT_ROOT / "experiments"
CHECKPOINTS = PROJECT_ROOT / "checkpoints"
KEYS = ["final_test_f1", "final_test_precision", "final_test_recall", "final_test_iou", "final_test_far"]


def _copy(src: Path, dst: Path) -> None:
    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def _seed_of(metrics_path: Path) -> int:
    return int(json.loads(metrics_path.read_text(encoding="utf-8"))["reproducibility"]["seed"])


def archive_seed42(name: str) -> None:
    base, archived = EXPERIMENTS / name, EXPERIMENTS / f"{name}_seed42"
    if (base / "metrics.json").exists() and not archived.exists() and _seed_of(base / "metrics.json") == 42:
        _copy(base, archived)
        _copy(CHECKPOINTS / name, CHECKPOINTS / f"{name}_seed42")
        print(f"Archived existing seed-42 run of {name}", flush=True)


def run_seed(script: str, name: str, seed: int, extra: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [sys.executable, str(PROJECT_ROOT / "src" / "training" / f"{script}.py"), "--seed", str(seed), *extra]
    print(f"\n=== {name} seed {seed} ===", flush=True)
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"{name} seed {seed} failed with exit code {result.returncode}")
    base = EXPERIMENTS / name
    # guard against a stale metrics.json being archived under the wrong seed
    if _seed_of(base / "metrics.json") != seed:
        raise RuntimeError(f"{name}: metrics.json seed does not match {seed}")
    _copy(base, EXPERIMENTS / f"{name}_seed{seed}")
    _copy(CHECKPOINTS / name, CHECKPOINTS / f"{name}_seed{seed}")


def summarize(name: str) -> dict:
    per_seed = {}
    for d in sorted(EXPERIMENTS.glob(f"{name}_seed*")):
        m = d / "metrics.json"
        if m.exists():
            doc = json.loads(m.read_text(encoding="utf-8"))
            per_seed[_seed_of(m)] = {k: doc[k] for k in KEYS} | {"validation_best_epoch": doc["validation_best_epoch"]}
    seeds = sorted(per_seed)
    summary = {"seeds": seeds, "n_seeds": len(seeds), "std_convention": "sample std (n-1)", "per_seed": per_seed}
    for k in KEYS:
        vals = [per_seed[s][k] for s in seeds if per_seed[s][k] is not None]
        summary[f"{k}_values"] = vals
        summary[f"{k}_mean"] = statistics.mean(vals) if vals else None
        summary[f"{k}_std"] = statistics.stdev(vals) if len(vals) > 1 else None
    (EXPERIMENTS / f"{name}_multiseed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{name}: F1 {summary['final_test_f1_values']}", flush=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--script", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    a = p.parse_args()
    archive_seed42(a.name)
    for s in a.seeds:
        run_seed(a.script, a.name, s, a.extra)
        summarize(a.name)


if __name__ == "__main__":
    main()
