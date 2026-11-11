"""Summarize OSCD Transformer test metrics at each run's validation-selected threshold (paper Table A3).

Reads FINAL_TUNING_RESULTS.csv (written by tuning_pipeline.py; thresholds were chosen on validation F1 only) and writes
threshold_selected_summary.json: per model, the seeds, the per-seed thresholds, and mean / sample SD of test F1 at 0.5 and
of test F1, precision, recall, IoU and FAR at the validation-selected threshold. Original checkpoints have seed 42 only.

Usage:
    .venv/Scripts/python.exe experiments/transformer_tuning_2026-10-15/summarize_threshold_selected.py
"""

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
METRICS = {"f1": "test_f1", "precision": "test_precision", "recall": "test_recall", "iou": "test_iou", "far": "test_fp_rate"}


def stat(vals):
    a = np.asarray(vals, float)
    return {"values": a.tolist(), "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else None}


def main() -> None:
    rows = [r for r in csv.DictReader(open(HERE / "FINAL_TUNING_RESULTS.csv", encoding="utf-8")) if r["dataset"] == "oscd"]
    out = {}
    for arch in ("self", "cross"):
        for ckpt in ("original", "tuned"):
            sel = sorted((r for r in rows if r["architecture"] == arch and r["comparison"] == f"{ckpt}_checkpoint_validation_threshold"),
                         key=lambda r: int(r["seed"]))
            at05 = {r["seed"]: r for r in rows if r["architecture"] == arch and r["comparison"] == f"{ckpt}_checkpoint_original_threshold"}
            out[f"{arch}_{ckpt}"] = {
                "seeds": [int(r["seed"]) for r in sel],
                "n": len(sel),
                "thresholds": [float(r["threshold"]) for r in sel],
                "test_f1_at_0_5": stat([float(at05[r["seed"]]["test_f1"]) for r in sel]),
                **{f"test_{k}_at_selected": stat([float(r[c]) for r in sel]) for k, c in METRICS.items()},
            }
    (HERE / "threshold_selected_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in out.items():
        print(k, v["n"], v["thresholds"], round(v["test_f1_at_0_5"]["mean"], 4), round(v["test_f1_at_selected"]["mean"], 4))


if __name__ == "__main__":
    main()
