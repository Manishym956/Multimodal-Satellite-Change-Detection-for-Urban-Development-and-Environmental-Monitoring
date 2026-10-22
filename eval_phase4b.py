"""Step 4B: evaluate the p4b_spatial checkpoints on the Step 4A-bis conditions (same evaluate(), same masks).

The comparison arm B (p2_dropout) was evaluated by eval_phase4abis.py with the identical function, so its stored
phase4abis counts are reused. Anchor: C's clean test counts at its stored clean threshold and at 0.5 equal the counts
in its own metrics.json (the single test evaluation at the end of training).
Also stores test counts inside / outside the true cloud mask M (eval_phase4abis_addendum.evaluate, the function used
for B's exploratory addendum); anchor: that pass's pooled counts equal this script's pooled test counts.

Usage:
    .venv/Scripts/python.exe eval_phase4b.py
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from eval_phase4abis import CONDITIONS, evaluate
from eval_phase4abis_addendum import evaluate as evaluate_inside_outside
from src.datasets.oscd_dataset import OSCDDataset
from src.phase3.corruption import load_spectrum
from src.training.train_robust import best_threshold, build_model, rates_at
from src.training.train_sar_baseline import split_train_cities

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiments" / "robust" / "phase4b"
ARM = "p4b_spatial"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    spectrum = torch.from_numpy(load_spectrum()).to(device)
    train_ds, test_ds = OSCDDataset(split="train", mode="fusion"), OSCDDataset(split="test", mode="fusion")
    _, val_cities = split_train_cities(train_ds.cities)
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    sources = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
               for p in ("eval_phase4b.py", "eval_phase4abis.py", "eval_phase4abis_addendum.py", "eval_phase4a.py",
                         "src/phase3/corruption.py", "src/training/train_robust.py")}
    failures = []
    for seed in range(1, 11):
        name = f"{ARM}_seed{seed}"
        if (OUT / "runs" / f"{name}.json").exists():
            print(name, "exists, skipped")
            continue
        ckpt = torch.load(ROOT / "checkpoints" / "robust" / f"{name}.pt", map_location="cpu", weights_only=False)
        model = build_model().to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        own = json.loads((ROOT / "experiments" / "robust" / name / "metrics.json").read_text(encoding="utf-8"))["test"]["full"]
        val = evaluate(model, train_ds, val_cities, spectrum, device, regions=False)
        test = evaluate(model, test_ds, list(test_ds.cities), spectrum, device, regions=True)
        io, io_pooled = evaluate_inside_outside(model, test_ds, spectrum, device)
        tau = ckpt["thresholds"]["full"]
        anchors = {"clean_val_argmax": best_threshold(val["pooled"]["clean"])[0] == tau,
                   **{f"clean_test_{k}": rates_at(test["pooled"]["clean"], kk) == {m: own[k][m] for m in rates_at(test["pooled"]["clean"], kk)}
                      for k, kk in (("at_selected", tau), ("at_0.5", 50))},
                   "inside_outside_pass_equals_pooled": all(torch.equal(io_pooled[c], test["pooled"][c]) for c in CONDITIONS),
                   "inside_plus_outside_equals_pooled": all(
                       torch.equal(io[f"inside__{k}"] + io[f"outside__{k}"], test["pooled"]["clean" if k.startswith("clean@") else k])
                       for k in (key[len("inside__"):] for key in io if key.startswith("inside__")))}
        if not all(anchors.values()):
            failures.append(name)
            print(name, "ANCHOR MISMATCH", anchors, flush=True)
            continue
        np.savez_compressed(OUT / "runs" / f"{name}.npz",
                            **{f"val__{c}": v.numpy() for c, v in val["pooled"].items()},
                            **{f"test__{c}": v.numpy() for c, v in test["pooled"].items()},
                            **{f"{key}__{c}": v.numpy() for key in ("missed", "over", "rest", "missed_clean", "over_clean", "rest_clean")
                               for c, v in test[key].items()},
                            **{f"city__{city}__{c}": v.numpy() for (city, c), v in test["city"].items()},
                            **{f"io_{k}": v.numpy() for k, v in io.items()})
        doc = {"run": name, "arm": ARM, "seed": seed, "tau_clean": tau / 100,
               "tau_level_val": {c: best_threshold(val["pooled"][c])[0] / 100 for c in CONDITIONS},
               "anchors": anchors, "source_sha256": sources}
        (OUT / "runs" / f"{name}.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
        print(name, " | ".join(f"{c} {rates_at(test['pooled'][c], tau)['f1']:.3f}" for c in ("clean", "c30", "z30", "c30__erode4", "c30__dilate4", "c30__miss25")), flush=True)
    if failures:
        sys.exit(1)
    print("P4B EVAL DONE")


if __name__ == "__main__":
    main()
