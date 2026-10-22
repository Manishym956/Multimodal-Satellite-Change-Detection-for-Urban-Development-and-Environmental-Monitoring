"""Score the Hafner-protocol checkpoints at DS-UNet's fixed config threshold (THRESH 0.66).

DS_UNet/configs/fusion_dualstream_*.yaml set THRESH: 0.66, which evaluation.py's
numeric_evaluation uses on the test set; train_network.py instead logs test F1 at the
threshold that maximises training-city F1. This script adds the 0.66 variant for the
final-step checkpoints. Nothing is selected from these numbers; it is one extra test
evaluation per checkpoint, recorded in experiments/robust/gap_analysis.json.

Usage:
    .venv/Scripts/python.exe gap_analysis_robust.py
"""

import json
import statistics
from datetime import datetime
from pathlib import Path

import torch

from src.datasets.oscd_dataset import OSCDDataset
from src.training.train_robust import build_model, evaluate, rates_at

ROOT = Path(__file__).resolve().parent
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_ds = OSCDDataset(split="test", mode="fusion")

rows = []
for path in sorted((ROOT / "experiments" / "robust").glob("hafner_full_seed*/metrics.json")):
    doc = json.loads(path.read_text(encoding="utf-8"))
    saved = torch.load(ROOT / "checkpoints" / "robust" / f"{path.parent.name}.pt", map_location="cpu", weights_only=False)
    model = build_model().to(device)
    model.load_state_dict(saved["model_state_dict"])
    counts = evaluate(model, test_ds, doc["test_cities"], device)["full"]
    rows.append({"run": path.parent.name, "seed": doc["args"]["seed"], "evaluated_at": datetime.now().isoformat(timespec="seconds"),
                 "f1_at_train_threshold": doc["test"]["full"]["at_selected"]["f1"],
                 "train_threshold": doc["test"]["full"]["selected_threshold"],
                 "f1_at_0.66": rates_at(counts, 66)["f1"], "f1_at_0.5": rates_at(counts, 50)["f1"]})
    print(rows[-1], flush=True)

agg = {k: {"mean": statistics.mean(r[k] for r in rows), "sd": statistics.stdev(r[k] for r in rows),
           "min": min(r[k] for r in rows), "max": max(r[k] for r in rows), "n": len(rows)}
       for k in ("f1_at_train_threshold", "f1_at_0.66", "f1_at_0.5")}
(ROOT / "experiments" / "robust" / "gap_analysis.json").write_text(json.dumps({"runs": rows, "aggregate": agg}, indent=2), encoding="utf-8")
print(json.dumps(agg, indent=2))
