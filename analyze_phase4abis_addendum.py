"""Step 4A-bis EXPLORATORY ADDENDUM analysis (not pre-registered; descriptive only, no hypothesis tests).

At each run's frozen clean-validation threshold, mean ± SD over 10 seeds:
  1. metrics inside the true cloud mask M and outside it (eval_phase4abis_addendum.py counts);
  2. pooled test IoU for every registered condition, from the registered phase4abis counts (IoU was not in the registered tables).

Writes experiments/robust/phase4abis/addendum_inside_outside/{summary.json, tables.md}.

Usage:
    .venv/Scripts/python.exe analyze_phase4abis_addendum.py
"""

import json
from pathlib import Path

import numpy as np
import torch

from analyze_phase3 import describe, ms
from src.phase3.mask_errors import VARIANTS
from src.training.train_robust import rates_at

P4ABIS = Path(__file__).resolve().parent / "experiments" / "robust" / "phase4abis"
OUT = P4ABIS / "addendum_inside_outside"
ARMS = {"p2_full": "A full-only", "p2_dropout": "B dropout"}
SEEDS = range(1, 11)
LEVELS = ("c10", "c30", "c60")
METRICS = ("f1", "precision", "recall", "false_alarm_rate")


def main() -> None:
    out = {"inside_outside": {}, "iou": {}}
    for arm in ARMS:
        docs = [json.loads((OUT / "runs" / f"{arm}_seed{s}.json").read_text(encoding="utf-8")) for s in SEEDS]
        assert all(all(d["anchors"].values()) for d in docs)
        add = [np.load(OUT / "runs" / f"{arm}_seed{s}.npz") for s in SEEDS]
        reg = [np.load(P4ABIS / "runs" / f"{arm}_seed{s}.npz") for s in SEEDS]
        tau = [round(d["tau_clean"] * 100) for d in docs]
        val = lambda arrs, key, m: [rates_at(torch.from_numpy(a[key]), k)[m] or 0.0 for a, k in zip(arrs, tau)]
        out["inside_outside"][arm] = {}
        for lv in LEVELS:
            for c in (f"clean@{lv}", lv, f"z{lv[1:]}", *(f"{lv}__{v}" for v in VARIANTS)):
                out["inside_outside"][arm][c] = {f"{part}_{m}": describe(val(add, f"{part}__{c}", m))
                                                 for part in ("inside", "outside") for m in METRICS}
        conds = ["clean", *LEVELS, "z10", "z30", "z60", *(f"{lv}__{v}" for lv in LEVELS for v in VARIANTS)]
        out["iou"][arm] = {c: describe(val(reg, f"test__{c}", "iou")) for c in conds}
    (OUT / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (OUT / "tables.md").write_text(tables(out), encoding="utf-8")
    print((OUT / "tables.md").read_text(encoding="utf-8"))


def tables(out: dict) -> str:
    L = ["### EXPLORATORY: metrics inside the true cloud mask M and outside it, frozen clean threshold (mean ± SD, 10 seeds)", "",
         "`clean@L` = clean input, scored on the pixels of level L's mask. Descriptive only; no tests.", "",
         "| arm | level | condition | F1 in M | recall in M | FAR in M | F1 outside M | recall outside M | FAR outside M |",
         "|---|---|---|---|---|---|---|---|---|"]
    for arm, name in ARMS.items():
        for c, e in out["inside_outside"][arm].items():
            lv = c[6:] if c.startswith("clean@") else (f"c{c[1:]}" if c.startswith("z") else c.split("__")[0])
            label = {"clean@": "clean", "z": "oracle fill"}.get(c[:6] if c.startswith("clean@") else c[:1], None)
            label = label or ("cloud, no fill" if c == lv else c.split("__")[1])
            L.append(f"| {name} | {lv[1:]}% | {label} | {ms(e['inside_f1'])} | {ms(e['inside_recall'])} | {ms(e['inside_false_alarm_rate'])} | "
                     f"{ms(e['outside_f1'])} | {ms(e['outside_recall'])} | {ms(e['outside_false_alarm_rate'])} |")
    L += ["", "### EXPLORATORY: pooled test IoU at the frozen clean threshold, from the registered counts (mean ± SD)", "",
          "| condition | A IoU | B IoU |", "|---|---|---|"]
    for c in out["iou"]["p2_full"]:
        L.append(f"| {c} | {ms(out['iou']['p2_full'][c])} | {ms(out['iou']['p2_dropout'][c])} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
