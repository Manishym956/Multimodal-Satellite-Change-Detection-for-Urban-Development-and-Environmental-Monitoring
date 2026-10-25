"""
Multi-seed runner for the clean asynchronous dual-stream CNN.
Runs seeds 43 and 44, saves results to seed-specific dirs,
then writes a combined summary JSON.

Usage:
    python run_multiseed_async_clean.py --seeds 43 44
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

EXPERIMENT_BASE = PROJECT_ROOT / "experiments" / "asynchronous_dual_stream_clean"
CHECKPOINT_BASE = PROJECT_ROOT / "checkpoints" / "asynchronous_dual_stream_clean"
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "training" / "train_asynchronous_dual_stream_clean.py"


def run_seed(seed: int) -> dict:
    seed_dir = EXPERIMENT_BASE.parent / f"asynchronous_dual_stream_clean_seed{seed}"
    ckpt_dir = CHECKPOINT_BASE.parent / f"asynchronous_dual_stream_clean_seed{seed}"

    print(f"\n{'='*60}", flush=True)
    print(f"Running seed {seed}...", flush=True)
    print(f"{'='*60}\n", flush=True)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--seed", str(seed)],
        cwd=str(PROJECT_ROOT),
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Training seed {seed} failed with exit code {result.returncode}")

    # Copy results to seed-specific directory
    if EXPERIMENT_BASE.exists():
        if seed_dir.exists():
            shutil.rmtree(seed_dir)
        shutil.copytree(EXPERIMENT_BASE, seed_dir)
        print(f"Saved experiment artifacts for seed {seed}", flush=True)

    if CHECKPOINT_BASE.exists():
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        shutil.copytree(CHECKPOINT_BASE, ckpt_dir)
        print(f"Saved checkpoint for seed {seed}", flush=True)

    metrics_path = seed_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(f"metrics.json not found at {metrics_path}")


def load_seed42() -> dict:
    path = EXPERIMENT_BASE / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Seed 42 metrics not found at {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def aggregate(all_metrics: dict) -> dict:
    """Compute mean ± std across seeds for key metrics."""
    import statistics

    keys = ["final_test_f1", "final_test_precision", "final_test_recall",
            "final_test_iou", "final_test_far"]

    seeds = sorted(all_metrics.keys())
    summary = {"seeds": seeds, "n_seeds": len(seeds), "per_seed": {}}
    for s in seeds:
        m = all_metrics[s]
        summary["per_seed"][s] = {k: m.get(k) for k in keys}

    for k in keys:
        vals = [all_metrics[s].get(k) for s in seeds if all_metrics[s].get(k) is not None]
        if len(vals) >= 2:
            summary[f"{k}_mean"] = statistics.mean(vals)
            summary[f"{k}_std"] = statistics.stdev(vals)
            summary[f"{k}_values"] = vals
        elif len(vals) == 1:
            summary[f"{k}_mean"] = vals[0]
            summary[f"{k}_std"] = 0.0
            summary[f"{k}_values"] = vals

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 44])
    args = parser.parse_args()

    # Load existing seed 42
    all_metrics = {42: load_seed42()}
    print(f"Loaded seed 42: F1={all_metrics[42]['final_test_f1']:.4f}", flush=True)

    for seed in args.seeds:
        all_metrics[seed] = run_seed(seed)
        f1 = all_metrics[seed]["final_test_f1"]
        print(f"Seed {seed} complete: F1={f1:.4f}", flush=True)

    # Write combined summary
    summary = aggregate(all_metrics)
    out_path = EXPERIMENT_BASE.parent / "asynchronous_dual_stream_clean_multiseed_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print("MULTI-SEED SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for k in ["final_test_f1", "final_test_precision", "final_test_recall",
              "final_test_iou", "final_test_far"]:
        mean = summary.get(f"{k}_mean")
        std = summary.get(f"{k}_std")
        vals = summary.get(f"{k}_values", [])
        if mean is not None:
            metric = k.replace("final_test_", "")
            print(f"  {metric:12s}: {mean:.4f} ± {std:.4f}  {[round(v,4) for v in vals]}")
    print(f"\nSaved summary to experiments/asynchronous_dual_stream_clean_multiseed_summary.json", flush=True)


if __name__ == "__main__":
    main()
