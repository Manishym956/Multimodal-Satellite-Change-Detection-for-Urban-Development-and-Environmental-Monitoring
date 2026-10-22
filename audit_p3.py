"""Independent audit of Phase 3: raw counts -> summary.json -> PHASE3_REPORT.md, masks -> manifest.

Recomputes from phase3/runs/*.npz with numpy only (not analyze_phase3.py code). Checks:
  masks: file hash = manifest, exact pixel count k = round(level*H*W), shape = city label shape
  runs:  20 present, anchors true, one source-hash set, eval_phase3.py hash = pre-registered (fixed) hash
  primary: F1 at the frozen clean threshold, per-seed values, mean, sample SD, paired diffs, 95% CI, t p,
           Holm over {c10,c30,c60,missing}, Wilcoxon; summary.json agrees
  report: every number in the PHASE3_REPORT.md primary table equals the recomputed value (3 decimals)
Writes experiments/robust/phase3/p3_audit.json; exits 1 on any failure.

Usage:
    .venv/Scripts/python.exe audit_p3.py
"""

import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

R = Path(__file__).resolve().parent / "experiments" / "robust"
P3 = R / "phase3"
LEVELS = {"c10": 0.10, "c30": 0.30, "c60": 0.60}
PRIMARY = ("c10", "c30", "c60", "missing")
EVAL_SHA = "5e28cf475e3f89c75601f11094bd817691c22b4f54273fd6a152f01368a673b3"
checks = []


def check(name, ok, detail=""):
    checks.append({"check": name, "pass": bool(ok), "detail": str(detail)[:300]})


def f1_at(counts, k):
    tp, fp, fn, _ = (int(v) for v in counts[k])
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def holm(p):
    adj, run = [0.0] * len(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = run
    return adj


def main():
    manifest = json.loads((P3 / "mask_manifest.json").read_text(encoding="utf-8"))
    check("manifest has 14 cities x 3 levels", len(manifest) == 42 and len({m["city"] for m in manifest}) == 14)
    check("val and test cities disjoint", not ({m["city"] for m in manifest if m["split"] == "val"} & {m["city"] for m in manifest if m["split"] == "test"}))
    for m in manifest:
        path = P3 / "masks" / m["file"]
        mask = np.load(path)
        h, w = mask.shape
        check(f"mask {m['file']}: hash", hashlib.sha256(path.read_bytes()).hexdigest() == m["sha256"])
        check(f"mask {m['file']}: exact fraction of image pixels",
              int(mask.sum()) == int(np.round(LEVELS[m["level"]] * h * w)) == m["k"] and [h, w] == m["shape"],
              f"{mask.sum()} vs {LEVELS[m['level']] * h * w}")

    runs = {}
    for arm in ("p2_full", "p2_dropout"):
        for s in range(1, 11):
            doc = json.loads((P3 / "runs" / f"{arm}_seed{s}.json").read_text(encoding="utf-8"))
            runs[(arm, s)] = (doc, np.load(P3 / "runs" / f"{arm}_seed{s}.npz"))
            check(f"{arm} seed {s}: anchors", all(doc["anchors"].values()))
            check(f"{arm} seed {s}: eval_phase3.py = pre-registered fixed hash", doc["source_sha256"]["eval_phase3.py"] == EVAL_SHA)
            p2 = json.loads((R / f"{arm}_seed{s}" / "metrics.json").read_text(encoding="utf-8"))["test"]
            check(f"{arm} seed {s}: frozen threshold = Phase 2 clean-validation threshold", doc["tau_clean"] == p2["full"]["selected_threshold"])
    check("one source-hash set", len({json.dumps(d["source_sha256"], sort_keys=True) for d, _ in runs.values()}) == 1)

    summ = json.loads((P3 / "summary.json").read_text(encoding="utf-8"))
    recomputed = {}
    for c in ("clean",) + PRIMARY:
        for arm in ("p2_full", "p2_dropout"):
            vals = [f1_at(runs[(arm, s)][1][f"test__{c}"], round(runs[(arm, s)][0]["tau_clean"] * 100)) for s in range(1, 11)]
            recomputed[(arm, c)] = vals
            st = summ["arms"][arm]["clean"][c]["f1"]
            check(f"summary {arm} {c}: per-seed F1 at frozen clean threshold, mean, sample SD",
                  np.allclose(st["values"], vals, atol=1e-12) and math.isclose(st["mean"], np.mean(vals), abs_tol=1e-12)
                  and math.isclose(st["sd"], np.std(vals, ddof=1), abs_tol=1e-12))
    pvals, rows = [], {}
    for c in PRIMARY:
        a, b = recomputed[("p2_full", c)], recomputed[("p2_dropout", c)]
        d = np.subtract(b, a)
        half = stats.t.ppf(0.975, 9) * d.std(ddof=1) / math.sqrt(10)
        rows[c] = {"mean": d.mean(), "ci": (d.mean() - half, d.mean() + half), "t_p": stats.ttest_rel(b, a).pvalue,
                   "w_p": stats.wilcoxon(b, a).pvalue, "n_pos": int((d > 0).sum()), "a": np.mean(a), "b": np.mean(b),
                   "a_sd": np.std(a, ddof=1), "b_sd": np.std(b, ddof=1)}
        pvals.append(rows[c]["t_p"])
    for c, h in zip(PRIMARY, holm(pvals)):
        rows[c]["holm"] = h
        r, s = rows[c], summ["paired"]["primary_clean"]["rows"][c]
        check(f"paired {c}: diffs/CI/t/Holm/Wilcoxon match summary (threshold = clean)",
              summ["paired"]["primary_clean"]["threshold"] == "clean" and math.isclose(s["mean_diff"], r["mean"], abs_tol=1e-12)
              and np.allclose(s["ci95"], r["ci"], atol=1e-12) and math.isclose(s["t_p"], r["t_p"], rel_tol=1e-9)
              and math.isclose(s["holm_p"], r["holm"], rel_tol=1e-9) and math.isclose(s["wilcoxon_p"], r["w_p"], rel_tol=1e-9))

    report = (R / "PHASE3_REPORT.md").read_text(encoding="utf-8")
    section = report.split("## 2.")[1].split("## 3.")[0]
    for c in PRIMARY:
        line = next((l for l in section.splitlines() if l.startswith(f"| {c} |")), "")
        nums = [float(x) for x in re.findall(r"[+-]?\d+\.\d+(?:e[+-]?\d+)?", line)]
        r = rows[c]
        expect = [r["a"], r["a_sd"], r["b"], r["b_sd"], r["mean"], r["ci"][0], r["ci"][1]]
        check(f"report primary row {c} matches recomputation", len(nums) >= 8
              and all(abs(x - y) <= 0.0005 + 1e-9 for x, y in zip(nums[:7], expect))
              and math.isclose(nums[7], r["holm"], rel_tol=0.06) and f"{r['n_pos']}/10" in line, line)
    prose = {"10% cloud cover costs A about 0.27 F1 and B about 0.22 F1": (
                 np.mean(np.subtract(recomputed[("p2_full", "c10")], recomputed[("p2_full", "clean")])),
                 np.mean(np.subtract(recomputed[("p2_dropout", "c10")], recomputed[("p2_dropout", "clean")])))}
    a_drop, b_drop = prose["10% cloud cover costs A about 0.27 F1 and B about 0.22 F1"]
    check("prose: c10 cost A≈0.27, B≈0.22", round(-a_drop, 2) == 0.27 and round(-b_drop, 2) == 0.22, (a_drop, b_drop))
    check("prose: B missing 0.478 vs c10 0.329", f"{np.mean(recomputed[('p2_dropout', 'missing')]):.3f}" == "0.478"
          and f"{np.mean(recomputed[('p2_dropout', 'c10')]):.3f}" == "0.329")
    out = {"n_checks": len(checks), "n_failed": sum(not c["pass"] for c in checks), "failed": [c for c in checks if not c["pass"]], "checks": checks}
    (P3 / "p3_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    assert np.allclose(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    main()
    failed = [c for c in checks if not c["pass"]]
    print(f"{len(checks)} checks, {len(failed)} failed")
    for c in failed:
        print("FAIL", c["check"], c["detail"])
    sys.exit(1 if failed else 0)
