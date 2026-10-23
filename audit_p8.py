"""Independent audit of Step 8 (written before any Step 8 run). numpy/scipy only, no analysis or evaluation code imported.

Checks: 40 evaluation files with true anchors and one source-hash set equal to the files on disk; the primary city set
re-derived from the scene manifest; every arm x tag metric; both slope tests; every family (Δ, CI, d_z, counts, p,
Holm, labels); WorldCover shares and the tested-class rule.
Writes experiments/robust/phase8/p8_audit.json; exit 1 on any failure.

Usage:
    .venv/Scripts/python.exe audit_p8.py
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiments" / "robust" / "phase8"
MANIFEST = ROOT / "datasets" / "oscd_s2_gee" / "manifest.json"
SEEDS = range(1, 11)
ARMS = ("p2_dropout", "p6_async", "p2_full", "p6_unet_sar")
TAGS = ("oscd", "gee_t1", "gap01", "gap03", "gap06", "gap12", "gap18")
GAPS = {"gap01": 1, "gap03": 3, "gap06": 6, "gap12": 12, "gap18": 18}
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
    for a in ARMS:
        for s in SEEDS:
            meta = json.loads((OUT / "runs" / f"{a}_seed{s}.json").read_text(encoding="utf-8"))
            runs[a, s] = (meta, np.load(OUT / "runs" / f"{a}_seed{s}.npz"))
            check(f"{a} {s} anchors", all(meta["anchors"].values()))
    hs = {json.dumps(runs[k][0]["source_sha256"], sort_keys=True) for k in runs}
    check("one source set", len(hs) == 1)
    for f, h in json.loads(hs.pop()).items():
        check(f"source unchanged {f}", hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == h)

    S = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    test = runs["p2_dropout", 1][0]["test_cities"]
    row = {(r["city"], r["tag"]): r for r in json.loads(MANIFEST.read_text(encoding="utf-8"))}
    primary = [c for c in test if all((c, t) in row and row[c, t].get("day") and row[c, t]["cloud"] <= 0.10 for t in GAPS)]
    check("primary cities", S["primary_cities"] == primary, (S["primary_cities"], primary))

    def pooled(a, s, tag, cities, cls=None):
        meta, z = runs[a, s]
        sar = a == "p6_unet_sar"
        tag = "oscd" if sar else tag
        k = meta["thresholds"]["full" if sar or tag == "full" else "no_o2"]
        return sum(z[f"{c}__{tag}__wc{cls}"] if cls is not None else z[f"{c}__{tag}"][k] for c in cities)

    met = lambda a, tag, k="f1", cities=primary, cls=None: np.array([rates(pooled(a, s, tag, cities, cls))[k] for s in SEEDS])
    for a in ARMS:
        for t in (TAGS if a != "p6_unet_sar" else ("oscd",)):
            for k in ("f1", "precision", "recall", "iou", "false_alarm_rate"):
                v, e = met(a, t, k), S["arms"][a][t][k]
                check(f"{a} {t} {k}", close(e["mean"], v.mean()) and close(e["sd"], v.std(ddof=1)))

    x = np.array(list(GAPS.values()), dtype=float)
    for key, arm in (("primary_rq2", "p2_dropout"), ("secondary_async_slope", "p6_async")):
        sl = np.array([np.polyfit(x, np.array([rates(pooled(arm, s, t, primary))["f1"] for t in GAPS]), 1)[0] for s in SEEDS])
        r = S[key]
        p = stats.ttest_1samp(sl, 0).pvalue
        check(f"{key} slopes", np.allclose(r["slopes_f1_per_month"], sl, atol=1e-12) and close(r["mean"], sl.mean())
              and close(r["t_p"], p, 1e-7) and close(r["wilcoxon_p"], stats.wilcoxon(sl).pvalue, 1e-7)
              and r["n_negative"] == int((sl < 0).sum()))
        want = "F1 falls with the gap" if p < 0.05 and sl.mean() < 0 else "F1 rises with the gap" if p < 0.05 and sl.mean() > 0 else "no detectable effect of the gap"
        check(f"{key} label", r["label"] == want, r["label"])

    z0 = runs["p6_unet_sar", 1][1]
    cnt = {cls: float(sum(z0[f"{c}__oscd__wc{cls}"].sum() for c in primary)) for cls in CLASSES}
    tot = sum(cnt.values())
    for cls in CLASSES:
        check(f"share {cls}", close(S["worldcover_share"][NAMES[cls]], cnt[cls] / tot))
    tested = [cls for cls in CLASSES if cls != 0 and cnt[cls] / tot >= 0.01]

    sar = met("p6_unet_sar", "oscd")
    fams = {"secondary_B_vs_sar": ({f"B@{t} - SAR-only": (sar, met("p2_dropout", t)) for t in GAPS}, "B"),
            "secondary_async_vs_sar": ({f"async@{t} - SAR-only": (sar, met("p6_async", t)) for t in GAPS}, "async"),
            "secondary_source": ({"B@gee_t1 - B@oscd": (met("p2_dropout", "oscd"), met("p2_dropout", "gee_t1")),
                                  "B@oscd - SAR-only": (sar, met("p2_dropout", "oscd"))}, None),
            "rq5_landcover": ({f"B@oscd - SAR-only, {NAMES[c]}": (met("p6_unet_sar", "oscd", "false_alarm_rate", primary, c),
                                                                 met("p2_dropout", "oscd", "false_alarm_rate", primary, c)) for c in tested}, "B")}
    for fam, (pairs, prefix) in fams.items():
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
            first = prefix or k.split(" - ")[0]
            want = f"{first} higher" if h < 0.05 and r["mean_diff"] > 0 else f"{first} lower" if h < 0.05 and r["mean_diff"] < 0 else "no detectable difference"
            check(f"{fam} {k} holm/label", close(r["holm_p"], h, 1e-7) and close(r["holm_wilcoxon_p"], hw, 1e-7) and r["label"] == want)

    bad = [r for r in res if not r["ok"]]
    (OUT / "p8_audit.json").write_text(json.dumps({"n_checks": len(res), "n_failed": len(bad), "failed": bad}, indent=2), encoding="utf-8")
    print(f"{len(res)} checks, {len(bad)} failed")
    for r in bad:
        print("FAIL", r["check"], r["detail"])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
