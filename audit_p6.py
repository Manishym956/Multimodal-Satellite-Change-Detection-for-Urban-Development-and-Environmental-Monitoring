"""Independent audit of Step 6 (written before any Step 6 run). numpy/scipy only, no analysis or training code imported.

Checks: 60 runs (6 arms x 10 seeds) with the registered settings, shared train/validation/test cities, test cities disjoint
from training and validation, one source-hash set per Step 6 arm equal to the files on disk, selected step = first
maximum of the stored validation score, F1 recomputed from raw confusion counts, every arm x condition x metric mean/SD,
every family's Δ, CI, d_z, counts, p-values, Holm and labels.
Writes experiments/robust/phase6/p6_audit.json; exit 1 on any failure.

Usage:
    .venv/Scripts/python.exe audit_p6.py
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
R = ROOT / "experiments" / "robust"
OUT = R / "phase6"
SEEDS = range(1, 11)
CONDS = ("full", "no_o2", "no_opt", "o2_copy")
REGISTERED = {  # arm: (arch, sar, train-conditions); None = Phase 2 run from train_robust.py
    "p2_full": None, "p2_dropout": None,
    "p6_unet_sar": ("unet_sar", "hafner_vv", "full=1"), "p6_unet_opt": ("unet_opt", "hafner_vv", "full=1"),
    "p6_unet_fusion": ("unet_fusion", "hafner_vv", "full=1"), "p6_async": ("dsunet", "hafner_vv", "no_o2=1")}
METRICS = ("f1", "precision", "recall", "iou", "false_alarm_rate")
res = []


def check(n, ok, d=""):
    res.append({"check": n, "ok": bool(ok), "detail": str(d)[:200]})


close = lambda a, b, t=1e-9: math.isclose(a, b, rel_tol=t, abs_tol=t)


def rates(e):
    tp, fp, fn, tn = (float(e[k]) for k in ("tp", "fp", "fn", "tn"))
    p, r = (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)
    return {"f1": 2 * p * r / (p + r) if p + r else 0.0, "precision": p, "recall": r,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0, "false_alarm_rate": fp / (fp + tn) if fp + tn else 0.0}


def holm(ps):
    order, adj, run = np.argsort(ps), [0.0] * len(ps), 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[j]))
        adj[j] = run
    return adj


def main():
    runs = {(a, s): json.loads((R / f"{a}_seed{s}" / "metrics.json").read_text(encoding="utf-8")) for a in REGISTERED for s in SEEDS}
    check("60 runs", len(runs) == 60)
    m0 = runs["p2_full", 1]
    for (a, s), m in runs.items():
        check(f"{a} {s} cities", (m["train_cities"], m["val_cities"], m["test_cities"]) == (m0["train_cities"], m0["val_cities"], m0["test_cities"]))
        check(f"{a} {s} steps", (m["args"]["steps"], m["args"]["eval_every"], m["args"]["seed"]) == (2500, 25, s))
        if REGISTERED[a]:
            arch, sar, tc = REGISTERED[a]
            check(f"{a} {s} registered settings", (m["args"].get("arch"), m["args"].get("sar"), m["args"]["train_conditions"]) == (arch, sar, tc))
            check(f"{a} {s} checkpoint exists", (ROOT / "checkpoints" / "robust" / f"{a}_seed{s}.pt").exists())
        scores = [h["val_score"] for h in m["history"]]
        check(f"{a} {s} selected step = first max", m["selected_step"] == m["history"][int(np.argmax(scores))]["step"])
        for c in CONDS:
            for thr in ("at_selected", "at_0.5"):
                e, rr = m["test"][c][thr], rates(m["test"][c][thr])
                check(f"{a} {s} {c} {thr} rates", all(close(e[k] or 0.0, rr[k]) for k in METRICS))
    check("test disjoint", not set(m0["test_cities"]) & set(m0["train_cities"] + m0["val_cities"]))
    check("val disjoint", not set(m0["val_cities"]) & set(m0["train_cities"]))
    for a in (k for k, v in REGISTERED.items() if v):
        hs = {json.dumps(runs[a, s]["source_sha256"], sort_keys=True) for s in SEEDS}
        check(f"{a} one source set", len(hs) == 1)
        disk = {"train_rrd.py": ROOT / "src/training/train_rrd.py", "train_robust.py": ROOT / "src/training/train_robust.py",
                "oscd_dataset.py": ROOT / "src/datasets/oscd_dataset.py",
                "networks.py": ROOT.parents[1] / "repos/2/DS_UNet/utils/networks.py",
                "loss_functions.py": ROOT.parents[1] / "repos/2/DS_UNet/utils/loss_functions.py"}
        for f, h in json.loads(hs.pop()).items():
            check(f"{a} source unchanged {f}", hashlib.sha256(disk[f].read_bytes()).hexdigest() == h)

    S = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    f1 = lambda a, c, k="f1", thr="at_selected": [rates(runs[a, s]["test"][c][thr])[k] for s in SEEDS]
    for a in REGISTERED:
        for thr in ("at_selected", "at_0.5"):
            for c in CONDS:
                for k in METRICS:
                    v, e = f1(a, c, k, thr), S["arms"][a][thr][c][k]
                    check(f"{a} {thr} {c} {k}", close(e["mean"], float(np.mean(v))) and close(e["sd"], float(np.std(v, ddof=1))))
    fams = {"primary_rq4": ({"A - SAR-only": (("p6_unet_sar", "full"), ("p2_full", "full")),
                             "A - optical-only": (("p6_unet_opt", "full"), ("p2_full", "full")),
                             "A - input fusion": (("p6_unet_fusion", "full"), ("p2_full", "full"))}, "DS-UNet"),
            "primary_rq1": ({"B@no_o2 - SAR-only": (("p6_unet_sar", "full"), ("p2_dropout", "no_o2")),
                             "async@no_o2 - SAR-only": (("p6_unet_sar", "full"), ("p6_async", "no_o2"))}, "SAR + historical optical"),
            "secondary_fusion": ({"fusion - optical-only": (("p6_unet_opt", "full"), ("p6_unet_fusion", "full")),
                                  "fusion - SAR-only": (("p6_unet_sar", "full"), ("p6_unet_fusion", "full")),
                                  "optical-only - SAR-only": (("p6_unet_sar", "full"), ("p6_unet_opt", "full"))}, None),
            "secondary_async": ({"async@no_o2 - B@no_o2": (("p2_dropout", "no_o2"), ("p6_async", "no_o2")),
                                 "B@no_o2 - B@no_opt": (("p2_dropout", "no_opt"), ("p2_dropout", "no_o2"))}, None)}
    for fam, (pairs, label) in fams.items():
        check(f"{fam} keys", list(S[fam]) == list(pairs))
        tps, wps = [], []
        for k, (x, y) in pairs.items():
            a, b = np.array(f1(*x)), np.array(f1(*y))
            d, r = b - a, S[fam][k]
            half = stats.t.ppf(0.975, 9) * d.std(ddof=1) / math.sqrt(10)
            # no variation: p = 1 (scipy cannot test it), d_z undefined
            tps.append(stats.ttest_rel(b, a).pvalue if d.std() else 1.0), wps.append(stats.wilcoxon(b, a).pvalue if d.any() else 1.0)
            dz_ok = close(r["dz"], d.mean() / d.std(ddof=1)) if d.std() else math.isnan(r["dz"])
            check(f"{fam} {k} Δ/CI/dz/count", close(r["mean_diff"], d.mean()) and close(r["ci95"][0], d.mean() - half)
                  and close(r["ci95"][1], d.mean() + half) and dz_ok and r["b_gt_a"] == int((d > 0).sum()))
            check(f"{fam} {k} p", close(r["t_p"], tps[-1], 1e-7) and close(r["wilcoxon_p"], wps[-1], 1e-7))
        for (k, r), h, hw in zip(S[fam].items(), holm(tps), holm(wps)):
            check(f"{fam} {k} holm", close(r["holm_p"], h, 1e-7) and close(r["holm_wilcoxon_p"], hw, 1e-7))
            if label:
                want = (f"{label} better" if h < 0.05 and r["mean_diff"] > 0 else f"{label} worse" if h < 0.05 and r["mean_diff"] < 0
                        else "no detectable difference")
                check(f"{fam} {k} label", r["label"] == want, r["label"])

    bad = [r for r in res if not r["ok"]]
    (OUT / "p6_audit.json").write_text(json.dumps({"n_checks": len(res), "n_failed": len(bad), "failed": bad}, indent=2), encoding="utf-8")
    print(f"{len(res)} checks, {len(bad)} failed")
    for r in bad:
        print("FAIL", r["check"], r["detail"])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
