"""Independent audit of Step 7 (written before any Step 7 run). numpy/scipy only, no analysis, evaluation or training code.

Checks: 60 evaluation files with true anchors and one source-hash set equal to the files on disk; the registered training
settings of every Step 7 run; per-city counts summing to the stored pooled counts; land-cover counts summing to the
full-input counts; every arm's 9-city and 10-city metrics; every family (Δ, CI, d_z, counts, p, Holm, labels); the
WorldCover pixel shares and the tested-class rule (>= 1%).
Writes experiments/robust/phase7/p7_audit.json; exit 1 on any failure.

Usage:
    .venv/Scripts/python.exe audit_p7.py
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
OUT = R / "phase7"
SEEDS = range(1, 11)
ARMS = {"p6_unet_sar": ("unet_sar", "hafner_vv", None), "p7_unet_vh": ("unet_sar", "vh", "full=1"),
        "p7_unet_vvvh": ("unet_sar", "vvvh", "full=1"), "p2_full": ("dsunet", "hafner_vv", None),
        "p7_dsunet_vh": ("dsunet", "vh", "full=1"), "p7_dsunet_vvvh": ("dsunet", "vvvh", "full=1")}
CITIES = ["brasilia", "montpellier", "norcia", "rio", "saclay_w", "valencia", "dubai", "milano", "chongqing"]
CLASSES = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)
NAMES = {0: "no data", 10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland", 50: "built-up", 60: "bare",
         70: "snow and ice", 80: "water", 90: "herbaceous wetland", 95: "mangroves", 100: "moss and lichen"}
res = []


def check(n, ok, d=""):
    res.append({"check": n, "ok": bool(ok), "detail": str(d)[:200]})


close = lambda a, b, t=1e-9: math.isclose(a, b, rel_tol=t, abs_tol=t)


def rates(c):
    tp, fp, fn, tn = (float(x) for x in c)
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
    runs = {}
    for a, (arch, sar, tc) in ARMS.items():
        for s in SEEDS:
            meta = json.loads((OUT / "runs" / f"{a}_seed{s}.json").read_text(encoding="utf-8"))
            z = np.load(OUT / "runs" / f"{a}_seed{s}.npz")
            mf = R / f"{a}_seed{s}" / "metrics.json"
            runs[a, s] = (meta, z)
            check(f"{a} {s} anchors", all(meta["anchors"].values()))
            check(f"{a} {s} arch/sar", (meta["arch"], meta["sar"]) == (arch, sar))
            check(f"{a} {s} source metrics.json exists", mf.exists())
            if not mf.exists():
                continue
            m = json.loads(mf.read_text(encoding="utf-8"))
            if tc:
                check(f"{a} {s} registered training", (m["args"]["arch"], m["args"]["sar"], m["args"]["train_conditions"],
                                                        m["args"]["steps"], m["args"]["eval_every"]) == (arch, sar, tc, 2500, 25))
            k = meta["thresholds"]["full"]
            pooled = sum(z[f"{c}__full"][k] for c in meta["test_cities"])
            e = m["test"]["full"]["at_selected"]
            check(f"{a} {s} pooled = stored", [int(x) for x in pooled] == [e["tp"], e["fp"], e["fn"], e["tn"]])
            for c in meta["test_cities"]:
                check(f"{a} {s} {c} land cover sums", np.array_equal(sum(z[f"{c}__wc{cls}"] for cls in CLASSES), z[f"{c}__full"][k]))
    hs = {json.dumps(runs[k][0]["source_sha256"], sort_keys=True) for k in runs}
    check("one source set", len(hs) == 1)
    for f, h in json.loads(hs.pop()).items():
        check(f"source unchanged {f}", hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == h)

    S = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    check("cities", S["cities"] == CITIES)
    pool = lambda a, s, cities: sum(runs[a, s][1][f"{c}__full"][runs[a, s][0]["thresholds"]["full"]] for c in cities)
    met = lambda a, k, cities=CITIES: np.array([rates(pool(a, s, cities))[k] for s in SEEDS])
    for a in ARMS:
        for scope, cities in (("9_cities", CITIES), ("10_cities", CITIES + ["lasvegas"])):
            for k in ("f1", "precision", "recall", "iou", "false_alarm_rate"):
                v, e = met(a, k, cities), S["arms"][a][scope][k]
                check(f"{a} {scope} {k}", close(e["mean"], v.mean()) and close(e["sd"], v.std(ddof=1)))
    z0 = runs["p6_unet_sar", 1][1]
    cnt = {cls: float(sum(z0[f"{c}__wc{cls}"].sum() for c in CITIES)) for cls in CLASSES}
    tot = sum(cnt.values())
    for cls in CLASSES:
        check(f"share {cls}", close(S["worldcover_share"][NAMES[cls]], cnt[cls] / tot))
    tested = [cls for cls in CLASSES if cls != 0 and cnt[cls] / tot >= 0.01]
    wc = lambda a, cls, k: np.array([rates(sum(runs[a, s][1][f"{c}__wc{cls}"] for c in CITIES))[k] for s in SEEDS])

    fams = {"primary_rq3": {"VV+VH - VV": (met("p6_unet_sar", "f1"), met("p7_unet_vvvh", "f1")),
                            "VH - VV": (met("p6_unet_sar", "f1"), met("p7_unet_vh", "f1"))},
            "secondary_dsunet": {"VV+VH - VV": (met("p2_full", "f1"), met("p7_dsunet_vvvh", "f1")),
                                 "VH - VV": (met("p2_full", "f1"), met("p7_dsunet_vh", "f1"))},
            "secondary_far": {f"VV+VH - VV, {NAMES[c]}": (wc("p6_unet_sar", c, "false_alarm_rate"), wc("p7_unet_vvvh", c, "false_alarm_rate"))
                              for c in tested}}
    for fam, pairs in fams.items():
        check(f"{fam} keys", list(S[fam]) == list(pairs))
        tps, wps = [], []
        for k, (a, b) in pairs.items():
            d, r = b - a, S[fam][k]
            half = stats.t.ppf(0.975, 9) * d.std(ddof=1) / math.sqrt(10)
            tps.append(stats.ttest_rel(b, a).pvalue if d.std() else 1.0), wps.append(stats.wilcoxon(b, a).pvalue if d.any() else 1.0)
            dz_ok = close(r["dz"], d.mean() / d.std(ddof=1)) if d.std() else math.isnan(r["dz"])
            check(f"{fam} {k} Δ/CI/dz/count", close(r["mean_diff"], d.mean()) and close(r["ci95"][0], d.mean() - half)
                  and close(r["ci95"][1], d.mean() + half) and dz_ok and r["b_gt_a"] == int((d > 0).sum()))
            check(f"{fam} {k} p", close(r["t_p"], tps[-1], 1e-7) and close(r["wilcoxon_p"], wps[-1], 1e-7))
        for (k, r), h, hw in zip(S[fam].items(), holm(tps), holm(wps)):
            first = k.split(" - ")[0]
            want = f"{first} higher" if h < 0.05 and r["mean_diff"] > 0 else f"{first} lower" if h < 0.05 and r["mean_diff"] < 0 else "no detectable difference"
            check(f"{fam} {k} holm/label", close(r["holm_p"], h, 1e-7) and close(r["holm_wilcoxon_p"], hw, 1e-7) and r["label"] == want)

    bad = [r for r in res if not r["ok"]]
    (OUT / "p7_audit.json").write_text(json.dumps({"n_checks": len(res), "n_failed": len(bad), "failed": bad}, indent=2), encoding="utf-8")
    print(f"{len(res)} checks, {len(bad)} failed")
    for r in bad:
        print("FAIL", r["check"], r["detail"])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
