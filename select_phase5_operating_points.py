"""Step 5: choose the detector operating points on VALIDATION composites only, from mask agreement (no change model is run).

Grid: probability threshold t in THRESHOLDS x dilation radius r in RADII (src/phase5/composite.py). For each (t, r), the
detector mask P = (prob >= t) dilated by a disk of radius r is compared with the true mask M on every validation composite
(all levels and replicates), and the pixel counts are pooled. Operating points:
    rb95, rb90, rb99   recall-biased: among points with pooled cloud recall >= 0.95 / 0.90 / 0.99, the one with the highest
                       pooled precision; ties: larger t, then smaller r. If no point reaches the target, the point with the
                       highest recall, flagged target_reached = false.
    bal                balanced: the highest pooled mask F1; ties: larger t, then smaller r.
Writes experiments/robust/phase5/operating_points.json (the points and the full validation grid). Refuses to overwrite.

Usage:
    .venv/Scripts/python.exe select_phase5_operating_points.py
"""

import json

import numpy as np

from src.phase5.composite import PHASE5, RADII, THRESHOLDS, detector_mask, mask_counts

TARGETS = {"rb95": 0.95, "rb90": 0.90, "rb99": 0.99}


def rates(c):
    tp, fp, fn, tn = (int(v) for v in c)
    rec, prec = tp / (tp + fn), (tp / (tp + fp) if tp + fp else 0.0)
    return {"recall": rec, "precision": prec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "overmasked_clear": fp / (fp + tn), "counts": [tp, fp, fn, tn]}


if __name__ == "__main__":
    dest = PHASE5 / "operating_points.json"
    if dest.exists():
        raise SystemExit("operating points exist; refusing to overwrite")
    grid = {(t, r): np.zeros(4, np.int64) for t in THRESHOLDS for r in RADII}
    names = sorted(f.stem for f in (PHASE5 / "detector_inputs" / "val").glob("*.npz") if not f.stem.endswith("_clean"))
    assert names, "no validation composites"
    for n in names:
        m = np.load(PHASE5 / "detector_inputs" / "val" / f"{n}.npz")["m"]
        prob = np.load(PHASE5 / "detector_outputs" / "val" / f"{n}.npz")["prob"]
        for t in THRESHOLDS:
            base = prob >= t
            for r in RADII:
                grid[(t, r)] += mask_counts(m, detector_mask(prob, t, r) if r else base)
    table = {(t, r): rates(c) for (t, r), c in grid.items()}
    order = lambda key: (-key[0], key[1])  # tie-break: larger t, then smaller r
    points = {}
    for name, target in TARGETS.items():
        ok = [k for k, v in table.items() if v["recall"] >= target]
        if ok:
            best = max(table[k]["precision"] for k in ok)
            k = sorted([k for k in ok if table[k]["precision"] == best], key=order)[0]
        else:
            best = max(v["recall"] for v in table.values())
            k = sorted([k for k, v in table.items() if v["recall"] == best], key=order)[0]
        points[name] = {"t": k[0], "r": k[1], "target_recall": target, "target_reached": bool(ok), **table[k]}
    best = max(v["f1"] for v in table.values())
    k = sorted([k for k, v in table.items() if v["f1"] == best], key=order)[0]
    points["bal"] = {"t": k[0], "r": k[1], **table[k]}
    doc = {"rule": __doc__, "validation_composites": names, "points": points,
           "grid": [{"t": t, "r": r, **v} for (t, r), v in table.items()]}
    dest.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    for n, p in points.items():
        print(n, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items() if k != "counts"})
