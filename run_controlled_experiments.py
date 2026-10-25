"""Run controlled comparison experiments.

Experiment 1 — Sync Dual-Stream CNN multi-seed (seeds 43 and 44)
  Script: src/training/train_dual_stream_baseline.py
  Purpose: Match the 3-seed evaluation already done for clean async CNN.
  Expected time: ~50 min per seed (~100 min total)

Experiment 2 — Transformer Cross-Attn with pos_weight=1 (seeds 42, 43, 44)
  Script: src/training/train_transformer_cross_attention.py --pos-weight 1
  Purpose: Controlled CNN vs Transformer comparison with matched loss function.
  Expected time: ~6 min per seed (~18 min total)

Usage:
    .venv/Scripts/python.exe run_controlled_experiments.py
    .venv/Scripts/python.exe run_controlled_experiments.py --skip-sync   # only Transformer
    .venv/Scripts/python.exe run_controlled_experiments.py --skip-xformer # only Sync CNN
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

SYNC_SCRIPT = PROJECT_ROOT / "src" / "training" / "train_dual_stream_baseline.py"
XFORMER_SCRIPT = PROJECT_ROOT / "src" / "training" / "train_transformer_cross_attention.py"

SYNC_EXP_BASE = PROJECT_ROOT / "experiments" / "dual_stream_baseline"
XFORMER_PW1_EXP_BASE = PROJECT_ROOT / "experiments" / "transformer_cross_attention_pw1"


def _make_env() -> dict:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run_seed(script: Path, seed: int, extra_args: list[str], label: str) -> float | None:
    """Run one training seed and return final test F1."""
    print(f"\n{'='*60}")
    print(f"Running {label} seed {seed}...")
    print(f"{'='*60}\n", flush=True)

    cmd = [PYTHON, str(script), "--seed", str(seed)] + extra_args
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=_make_env())

    if result.returncode != 0:
        print(f"ERROR: {label} seed {seed} exited with code {result.returncode}", flush=True)
        return None

    print(f"Seed {seed} complete.", flush=True)
    return None  # F1 read from metrics.json after run


def _read_f1(metrics_path: Path) -> float | None:
    if not metrics_path.exists():
        return None
    with metrics_path.open(encoding="utf-8") as fh:
        doc = json.load(fh)
    return doc.get("final_test_f1")


def _save_seed_artifacts(exp_base: Path, seed: int) -> None:
    """Copy experiment dir to seed-specific subfolder for archival."""
    seed_dir = exp_base.parent / f"{exp_base.name}_seed{seed}"
    if seed_dir.exists():
        shutil.rmtree(seed_dir)
    shutil.copytree(exp_base, seed_dir)
    print(f"Saved experiment artifacts for seed {seed}", flush=True)

    # Also copy checkpoint
    ckpt_base = PROJECT_ROOT / "checkpoints" / exp_base.name
    ckpt_seed = PROJECT_ROOT / "checkpoints" / f"{exp_base.name}_seed{seed}"
    if ckpt_base.exists():
        if ckpt_seed.exists():
            shutil.rmtree(ckpt_seed)
        shutil.copytree(ckpt_base, ckpt_seed)
        print(f"Saved checkpoint for seed {seed}", flush=True)


def _print_summary(label: str, seeds: list[int], f1s: list[float]) -> None:
    print(f"\n{'='*60}")
    print(f"SUMMARY: {label}")
    print(f"{'='*60}")
    for s, f in zip(seeds, f1s):
        print(f"  Seed {s}: F1 = {f:.4f}")
    if len(f1s) >= 2:
        mean = statistics.mean(f1s)
        std = statistics.stdev(f1s) if len(f1s) > 1 else 0.0
        print(f"  Mean: {mean:.4f}  Std: {std:.4f}  Values: {[round(f,4) for f in f1s]}")
    print(flush=True)


def run_sync_cnn_multiseed() -> None:
    """Run sync dual-stream CNN for seeds 43 and 44 (seed 42 already done)."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Sync Dual-Stream CNN — Seeds 43, 44")
    print("="*60)

    # Read existing seed 42 result
    f1_42 = _read_f1(SYNC_EXP_BASE / "metrics.json")
    if f1_42 is not None:
        print(f"Loaded seed 42: F1={f1_42:.4f}", flush=True)
    else:
        print("WARNING: seed 42 metrics.json not found; will report 43+44 only")
        f1_42 = None

    results: dict[int, float] = {}
    if f1_42 is not None:
        results[42] = f1_42

    for seed in [43, 44]:
        t0 = time.perf_counter()
        _run_seed(SYNC_SCRIPT, seed, [], f"Sync Dual-Stream CNN")
        elapsed = time.perf_counter() - t0
        f1 = _read_f1(SYNC_EXP_BASE / "metrics.json")
        if f1 is not None:
            results[seed] = f1
            _save_seed_artifacts(SYNC_EXP_BASE, seed)
            print(f"Seed {seed} complete: F1={f1:.4f}  ({elapsed:.0f}s)", flush=True)
        else:
            print(f"WARNING: could not read F1 for seed {seed}", flush=True)

    if len(results) >= 2:
        seeds_done = sorted(results.keys())
        f1s = [results[s] for s in seeds_done]
        _print_summary("Sync Dual-Stream CNN", seeds_done, f1s)

        # Save multi-seed summary
        summary = {
            "seeds": seeds_done,
            "final_test_f1_values": f1s,
            "final_test_f1_mean": round(statistics.mean(f1s), 4),
            "final_test_f1_std": round(statistics.stdev(f1s) if len(f1s) > 1 else 0.0, 4),
        }
        summary_path = PROJECT_ROOT / "experiments" / "dual_stream_baseline_multiseed_summary.json"
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print("Saved multi-seed summary.", flush=True)


def run_transformer_pw1_multiseed() -> None:
    """Run Transformer Cross-Attn with pos_weight=1 for seeds 42, 43, 44."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Transformer Cross-Attn pos_weight=1 — Seeds 42, 43, 44")
    print("="*60)

    extra = ["--pos-weight", "1"]
    results: dict[int, float] = {}

    for seed in [42, 43, 44]:
        t0 = time.perf_counter()
        _run_seed(XFORMER_SCRIPT, seed, extra, "Transformer Cross-Attn pw=1")
        elapsed = time.perf_counter() - t0
        f1 = _read_f1(XFORMER_PW1_EXP_BASE / "metrics.json")
        if f1 is not None:
            results[seed] = f1
            _save_seed_artifacts(XFORMER_PW1_EXP_BASE, seed)
            print(f"Seed {seed} complete: F1={f1:.4f}  ({elapsed:.0f}s)", flush=True)
        else:
            print(f"WARNING: could not read F1 for seed {seed}", flush=True)

    if len(results) >= 2:
        seeds_done = sorted(results.keys())
        f1s = [results[s] for s in seeds_done]
        _print_summary("Transformer Cross-Attn pw=1", seeds_done, f1s)

        summary = {
            "seeds": seeds_done,
            "final_test_f1_values": f1s,
            "final_test_f1_mean": round(statistics.mean(f1s), 4),
            "final_test_f1_std": round(statistics.stdev(f1s) if len(f1s) > 1 else 0.0, 4),
        }
        summary_path = PROJECT_ROOT / "experiments" / "transformer_cross_attention_pw1_multiseed_summary.json"
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print("Saved multi-seed summary.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-sync",    action="store_true", help="Skip sync CNN multi-seed")
    parser.add_argument("--skip-xformer", action="store_true", help="Skip Transformer pw=1")
    args = parser.parse_args()

    t_start = time.perf_counter()

    if not args.skip_sync:
        run_sync_cnn_multiseed()

    if not args.skip_xformer:
        run_transformer_pw1_multiseed()

    total = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"ALL CONTROLLED EXPERIMENTS COMPLETE in {total/60:.1f} min")
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
