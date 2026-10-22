"""Re-evaluate finished train_robust runs from their checkpoints, without touching metrics.json.

For every run, the saved (selected) checkpoint is evaluated once on its own
validation cities and once on the test cities. Output: <run>/reeval.json with

  reproduction  for every condition already in metrics.json: stored vs re-evaluated
                TP/FP/FN/TN at the stored threshold and at 0.5. Tolerance: the four
                counts must match exactly (evaluation is deterministic: cudnn
                deterministic, no randomness, first orbit per city).
  added         conditions missing from metrics.json, each with a threshold chosen on
                the validation cities only (same rule as training) and test metrics.

Exit code 1 if any count differs. Runs that already have reeval.json are skipped.

Usage:
    .venv/Scripts/python.exe reeval_robust.py
"""

import json
import sys
from pathlib import Path

import torch

from src.datasets.oscd_dataset import OSCDDataset
from src.training.train_robust import CONDITIONS, best_threshold, build_model, evaluate, rates_at

ROOT = Path(__file__).resolve().parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False  # match training-time evaluation
train_ds, test_ds = OSCDDataset(split="train", mode="fusion"), OSCDDataset(split="test", mode="fusion")
COUNTS = ("tp", "fp", "fn", "tn")
failures = 0

for path in sorted((ROOT / "experiments" / "robust").glob("*_seed*/metrics.json")):
    out = path.parent / "reeval.json"
    if out.exists():
        continue
    doc = json.loads(path.read_text(encoding="utf-8"))
    saved = torch.load(ROOT / "checkpoints" / "robust" / f"{path.parent.name}.pt", map_location="cpu", weights_only=False)
    if saved["step"] != doc["selected_step"]:
        raise SystemExit(f"{path.parent.name}: checkpoint step {saved['step']} != selected step {doc['selected_step']}")
    model = build_model().to(device)
    model.load_state_dict(saved["model_state_dict"])
    val = evaluate(model, train_ds, doc["val_cities"], device)
    test = evaluate(model, test_ds, doc["test_cities"], device)

    reproduction, added = {}, {}
    for c, stored in doc["test"].items():
        k = round(stored["selected_threshold"] * 100)
        for at, kk in (("at_selected", k), ("at_0.5", 50)):
            new = rates_at(test[c], kk)
            diff = {n: new[n] - stored[at][n] for n in COUNTS}
            reproduction[f"{c}/{at}"] = {"count_diff": diff, "exact": not any(diff.values())}
            failures += any(diff.values())
        # the stored threshold must also be the validation argmax for this checkpoint
        reproduction[f"{c}/threshold_from_val"] = best_threshold(val[c])[0] == k
        failures += best_threshold(val[c])[0] != k
    for c in CONDITIONS:
        if c not in doc["test"]:
            k = best_threshold(val[c])[0]
            added[c] = {"selected_threshold": k / 100, "at_selected": rates_at(test[c], k), "at_0.5": rates_at(test[c], 50)}
    out.write_text(json.dumps({"checkpoint_step": saved["step"], "reproduction": reproduction, "added": added}, indent=2), encoding="utf-8")
    ok = all(v if isinstance(v, bool) else v["exact"] for v in reproduction.values())
    print(path.parent.name, "reproduced exactly" if ok else "MISMATCH", "| added", list(added), flush=True)

sys.exit(1 if failures else 0)
