"""Independent audit of Step 5S (written before eval_phase5s.py ran). numpy/scipy only, no analysis code imported.

Checks: 30 runs with true anchors and one source-hash set equal to the files on disk; r* re-derived from validation counts;
every arm x condition mean/SD; every family's Δ, CI, p-values and Holm; primary labels; retained gain.
Writes experiments/robust/phase5s/p5s_audit.json; exit 1 on any failure.

Usage:
    .venv/Scripts/python.exe audit_p5s.py
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiments" / "robust" / "phase5s"
ARMS, SEEDS, LEVELS = ("p2_full", "p2_dropout", "p4b_spatial"), range(1, 11), ("c10", "c30", "c60")
UNDER, RADII = ("erode2", "erode4", "erode8", "miss10", "miss25", "miss50"), (0, 1, 2, 3, 4, 6, 8, 12, 16)
res = []


def check(n, ok, d=""):
    res.append({"check": n, "ok": bool(ok), "detail": str(d)[:200]})


close = lambda a, b, t=1e-9: math.isclose(a, b, rel_tol=t, abs_tol=t)


def f1c(c, k):
    tp, fp, fn, _ = (float(x) for x in c[k])
    p, r = (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)
    return 2 * p * r / (p + r) if p + r else 0.0


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
            d = json.loads((OUT / "runs" / f"{a}_seed{s}.json").read_text(encoding="utf-8"))
            check(f"{a} {s} anchors", all(d["anchors"].values()))
            runs[(a, s)] = (d, np.load(OUT / "runs" / f"{a}_seed{s}.npz"))
    hs = {json.dumps(d["source_sha256"], sort_keys=True) for d, _ in runs.values()}
    check("one source set", len(hs) == 1)
    for f, h in json.loads(hs.pop()).items():
        check(f"source unchanged {f}", hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == h)
    f1 = lambda a, c, sp="test": [f1c(runs[(a, s)][1][f"{sp}__{c}"], round(runs[(a, s)][0]["tau_clean"] * 100)) for s in SEEDS]
    S = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    vs = {r: np.mean([np.mean(f1("p2_dropout", f"{lv}__{v}__{r}", "val")) for lv in LEVELS for v in UNDER]) for r in RADII}
    rstar = min(r for r in RADII if vs[r] == max(vs.values()))
    check("r* re-derived", S["radius_selection"]["r_star"] == rstar, (rstar, S["radius_selection"]["r_star"]))
    for a in ARMS:
        for c, e in S["arms"][a].items():
            v = f1(a, c)
            check(f"{a} {c}", close(e["mean"], float(np.mean(v))) and close(e["sd"], float(np.std(v, ddof=1))))

    def fam(name, pairs):
        rows = S[name]
        check(f"{name} keys", set(rows) == set(pairs))
        tp, wp = {}, {}
        for k, (x, y) in pairs.items():
            d = np.array(y) - np.array(x)
            sd = d.std(ddof=1)
            h = stats.t.ppf(0.975, 9) * sd / math.sqrt(10)
            tp[k] = float(stats.ttest_rel(y, x).pvalue) if sd else 1.0
            wp[k] = float(stats.wilcoxon(y, x).pvalue) if d.any() else 1.0
            r = rows[k]
            check(f"{name} {k}", close(r["mean_diff"], d.mean()) and close(r["ci95"][0], d.mean() - h) and close(r["ci95"][1], d.mean() + h)
                  and r["b_gt_a"] == int((d > 0).sum()) and close(r["t_p"], tp[k], 1e-7) and close(r["wilcoxon_p"], wp[k], 1e-7))
        for k, hh in zip(tp, holm(list(tp.values()))):
            check(f"{name} {k} holm", close(rows[k]["holm_p"], hh, 1e-7))
        for k, hh in zip(wp, holm(list(wp.values()))):
            check(f"{name} {k} holm W", close(rows[k]["holm_wilcoxon_p"], hh, 1e-7))

    dil = lambda a, lvs: {f"{lv}__{v}": (f1(a, f"{lv}__{v}__0"), f1(a, f"{lv}__{v}__{rstar}")) for lv in lvs for v in UNDER}
    fam("primary", dil("p2_dropout", ("c30",)))
    fam("secondary_B_c10_c60", dil("p2_dropout", ("c10", "c60")))
    fam("secondary_A", dil("p2_full", LEVELS))
    fam("secondary_C", dil("p4b_spatial", LEVELS))
    fam("secondary_B_vs_oracle", {f"{lv}__{v}": (f1("p2_dropout", f"z{lv[1:]}"), f1("p2_dropout", f"{lv}__{v}__{rstar}")) for lv in LEVELS for v in UNDER})
    for k, r in S["primary"].items():
        want = ("dilation better" if r["holm_p"] < 0.05 and r["mean_diff"] > 0 else "dilation worse" if r["holm_p"] < 0.05 and r["mean_diff"] < 0
                else "no detectable difference")
        check(f"label {k}", r["label"] == want)
    for a in ARMS:
        for lv in LEVELS:
            for v in UNDER:
                c, o, x = (np.array(f1(a, k)) for k in (lv, f"z{lv[1:]}", f"{lv}__{v}__{rstar}"))
                check(f"retained {a} {lv} {v}", close(S["retained_gain"][a][f"{lv}__{v}"]["r_star"]["mean"], float(((x - c) / (o - c)).mean())))
    bad = [r for r in res if not r["ok"]]
    (OUT / "p5s_audit.json").write_text(json.dumps({"n_checks": len(res), "n_failed": len(bad), "failed": bad}, indent=2), encoding="utf-8")
    print(f"{len(res)} checks, {len(bad)} failed")
    for r in bad:
        print("FAIL", r["check"], r["detail"])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
