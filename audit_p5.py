"""Independent audit of Step 5 (written before any Step 5 test evaluation).

Recomputes from raw files with numpy/scipy only, without importing the analysis code:
  inputs:    composite files against the manifest; detector outputs against detector_info; operating points re-derived from
             the validation detector outputs with an independent implementation of the selection rule
  runs:      30 run files, all anchors true, one source-hash set, sources equal to the files on disk
  metrics:   every arm x condition x metric mean / SD at the frozen clean threshold, from the raw counts
  families:  every contrast (Δ, CI, t p, Wilcoxon p, Holm) of every family, and the primary labels
  derived:   retained gain; detector masks against M on the test composites
  report:    if PHASE5_REPORT.md exists, every number in its primary table equals the recomputation
Writes experiments/robust/phase5/p5_audit.json; exit 1 on any failure.

Usage:
    .venv/Scripts/python.exe audit_p5.py
"""

import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage, stats

ROOT = Path(__file__).resolve().parent
P5 = ROOT / "experiments" / "robust" / "phase5"
SEEDS = range(1, 11)
ARMS = ("p2_full", "p2_dropout", "p4b_spatial")
LEVELS = ("c10", "c30", "c60")
DETS = ("rb95", "rb90", "rb99", "bal", "default")
results = []


def check(name, ok, detail=""):
    results.append({"check": name, "ok": bool(ok), "detail": str(detail)[:300]})


def close(a, b, tol=1e-9):
    return (a is None and b is None) or (a is not None and b is not None and (math.isclose(a, b, rel_tol=tol, abs_tol=tol)
                                                                              or (math.isnan(a) and math.isnan(b))))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rates(c, k):
    tp, fp, fn, tn = (float(x) for x in c[k])
    p, r = (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)
    return {"f1": 2 * p * r / (p + r) if p + r else 0.0, "precision": p, "recall": r,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0, "false_alarm_rate": fp / (fp + tn) if fp + tn else 0.0}


def holm(ps):
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[j]))
        adj[j] = run
    return adj


def disk_dilate(m, r):
    if r == 0:
        return m
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return ndimage.binary_dilation(m, x * x + y * y <= r * r)


def main() -> None:
    man = json.loads((P5 / "composite_manifest.json").read_text(encoding="utf-8"))
    check("composite files match manifest", all(sha(P5 / "composites" / e["file"]) == e["sha256"] for e in man["entries"]))
    for split in ("val", "test"):
        info = json.loads((P5 / "detector_outputs" / split / "detector_info.json").read_text(encoding="utf-8"))
        check(f"detector {split} inputs/outputs match detector_info", all(
            sha(P5 / "detector_inputs" / split / f) == v["input_sha256"] and sha(P5 / "detector_outputs" / split / f) == v["output_sha256"]
            for f, v in info["files"].items()))
    # operating points, independent re-derivation
    ops = json.loads((P5 / "operating_points.json").read_text(encoding="utf-8"))
    ts, rs = [round(0.05 * i, 2) for i in range(1, 19)], [0, 1, 2, 3, 4, 6, 8, 12, 16]
    grid = {(t, r): np.zeros(4) for t in ts for r in rs}
    for n in ops["validation_composites"]:
        m = np.load(P5 / "detector_inputs" / "val" / f"{n}.npz")["m"]
        p = np.load(P5 / "detector_outputs" / "val" / f"{n}.npz")["prob"]
        for t in ts:
            for r in rs:
                q = disk_dilate(p >= t, r)
                grid[(t, r)] += [(m & q).sum(), (~m & q).sum(), (m & ~q).sum(), (~m & ~q).sum()]
    rec = {k: v[0] / (v[0] + v[2]) for k, v in grid.items()}
    prec = {k: (v[0] / (v[0] + v[1]) if v[0] + v[1] else 0.0) for k, v in grid.items()}
    f1m = {k: (2 * prec[k] * rec[k] / (prec[k] + rec[k]) if prec[k] + rec[k] else 0.0) for k in grid}
    key = lambda k: (-k[0], k[1])
    for name, target in (("rb95", 0.95), ("rb90", 0.90), ("rb99", 0.99)):
        ok = [k for k in grid if rec[k] >= target]
        pool, score = (ok, prec) if ok else (list(grid), rec)
        best = max(score[k] for k in pool)
        k = sorted([k for k in pool if score[k] == best], key=key)[0]
        check(f"operating point {name} re-derived", (ops["points"][name]["t"], ops["points"][name]["r"]) == k, (k, ops["points"][name]))
    best = max(f1m.values())
    k = sorted([k for k in grid if f1m[k] == best], key=key)[0]
    check("operating point bal re-derived", (ops["points"]["bal"]["t"], ops["points"]["bal"]["r"]) == k)

    # runs
    runs = {}
    for arm in ARMS:
        for s in SEEDS:
            doc = json.loads((P5 / "runs" / f"{arm}_seed{s}.json").read_text(encoding="utf-8"))
            check(f"{arm} seed {s} anchors", all(doc["anchors"].values()), doc["anchors"])
            runs[(arm, s)] = (doc, np.load(P5 / "runs" / f"{arm}_seed{s}.npz"))
    hs = {json.dumps(d["source_sha256"], sort_keys=True) for d, _ in runs.values()}
    check("one source-hash set", len(hs) == 1)
    for f, h in json.loads(hs.pop()).items():
        path = ROOT / f if (ROOT / f).exists() else P5 / "detector_outputs" / f.split("_")[-1].split(".")[0] / "detector_info.json"
        check(f"source unchanged: {f}", sha(path) == h)
    summ = json.loads((P5 / "summary.json").read_text(encoding="utf-8"))
    f1 = lambda arm, c: [rates(runs[(arm, s)][1][f"test__{c}"], round(runs[(arm, s)][0]["tau_clean"] * 100))["f1"] for s in SEEDS]
    for arm in ARMS:
        for c, e in summ["arms"][arm].items():
            for mt, v in e.items():
                vals = [rates(runs[(arm, s)][1][f"test__{c}"], round(runs[(arm, s)][0]["tau_clean"] * 100))[mt] for s in SEEDS]
                check(f"{arm} {c} {mt} mean/sd", close(v["mean"], float(np.mean(vals))) and close(v["sd"], float(np.std(vals, ddof=1))))

    def fam(name, pairs):
        rows = summ[name]
        check(f"{name}: same contrasts", set(rows) == set(pairs), (sorted(rows), sorted(pairs)))
        tp, wp = {}, {}
        for k, (a, b) in pairs.items():
            d = np.array(b) - np.array(a)
            sd = d.std(ddof=1)
            half = stats.t.ppf(0.975, 9) * sd / math.sqrt(10)
            tp[k] = float(stats.ttest_rel(b, a).pvalue) if sd else 1.0
            wp[k] = float(stats.wilcoxon(b, a).pvalue) if d.any() else 1.0
            r = rows[k]
            check(f"{name} {k} Δ/CI/count", close(r["mean_diff"], d.mean()) and close(r["ci95"][0], d.mean() - half)
                  and close(r["ci95"][1], d.mean() + half) and r["b_gt_a"] == int((d > 0).sum()))
            check(f"{name} {k} p-values", close(r["t_p"], tp[k], 1e-7) and close(r["wilcoxon_p"], wp[k], 1e-7))
        for k, h in zip(tp, holm(list(tp.values()))):
            check(f"{name} {k} Holm t", close(rows[k]["holm_p"], h, 1e-7))
        for k, h in zip(wp, holm(list(wp.values()))):
            check(f"{name} {k} Holm W", close(rows[k]["holm_wilcoxon_p"], h, 1e-7))

    contrasts = lambda arm: {f"{lv}: {b} - {a}": (f1(arm, f"{lv}__{a}"), f1(arm, f"{lv}__{b}"))
                             for lv in LEVELS for a, b in (("bal", "rb95"), ("oracle", "rb95"), ("cloud", "rb95"))}
    fam("primary", contrasts("p2_dropout"))
    fam("secondary_A", contrasts("p2_full"))
    fam("secondary_C", contrasts("p4b_spatial"))
    fam("secondary_B_other_masks", {f"{lv}: {d} - bal": (f1("p2_dropout", f"{lv}__bal"), f1("p2_dropout", f"{lv}__{d}"))
                                    for lv in LEVELS for d in ("rb90", "rb99", "default")})
    fam("secondary_C_minus_B", {c: (f1("p2_dropout", c), f1("p4b_spatial", c)) for c in ("clean", "c10__rb95", "c30__rb95", "c60__rb95")})
    fam("secondary_B_false_detections", {f"clean__{d} - clean": (f1("p2_dropout", "clean"), f1("p2_dropout", f"clean__{d}")) for d in DETS})
    for k, r in summ["primary"].items():
        want = ("rb95 better" if r["holm_p"] < 0.05 and r["mean_diff"] > 0 else "rb95 worse" if r["holm_p"] < 0.05 and r["mean_diff"] < 0
                else "no detectable difference")
        check(f"primary label {k}", r["label"] == want)
    for arm in ARMS:
        for lv in LEVELS:
            for d in DETS:
                c, o, x = (np.array(f1(arm, f"{lv}__{k}")) for k in ("cloud", "oracle", d))
                g = (x - c) / (o - c)
                check(f"retained {arm} {lv} {d}", close(summ["retained_gain"][arm][lv][d]["mean"], float(g.mean())))
    nonbank = {lv: [e["file"][:-4] for e in man["entries"] if e["split"] == "test" and e["level"] == lv
                    and e["scene"] not in set(man["exclusions_and_flags"]["flag_bank"])] for lv in LEVELS}
    for lv in LEVELS:
        r = summ["C_minus_B_non_bank_composites"][lv]
        check(f"non-bank composite count {lv}", r["n_composites"] == len(nonbank[lv]))
        for cond in (("cloud", "oracle", "rb95", "bal") if nonbank[lv] else ()):
            d = []
            for s in SEEDS:
                v = {}
                for arm in ("p2_dropout", "p4b_spatial"):
                    doc, arr = runs[(arm, s)]
                    v[arm] = rates(sum(arr[f"comp__{n}__{lv}__{cond}"] for n in nonbank[lv]), round(doc["tau_clean"] * 100))["f1"]
                d.append(v["p4b_spatial"] - v["p2_dropout"])
            check(f"non-bank C - B {lv} {cond}", close(r[cond]["mean"], float(np.mean(d))))
    for lv in LEVELS:
        for d in DETS:
            cnt = np.zeros(4)
            for f in sorted((P5 / "detector_inputs" / "test").glob(f"*_{lv}_k*.npz")):
                m = np.load(f)["m"]
                o = np.load(P5 / "detector_outputs" / "test" / f.name)
                q = o["default"] if d == "default" else disk_dilate(o["prob"] >= ops["points"][d]["t"], ops["points"][d]["r"])
                cnt += [(m & q).sum(), (~m & q).sum(), (m & ~q).sum(), (~m & ~q).sum()]
            r = summ["detector_test"][lv][d]
            check(f"detector test {lv} {d}", close(r["recall"], cnt[0] / (cnt[0] + cnt[2])) and close(r["precision"], cnt[0] / (cnt[0] + cnt[1])))
    rep = ROOT / "experiments" / "robust" / "PHASE5_REPORT.md"
    if rep.exists():
        sec = rep.read_text(encoding="utf-8").split("## 2.")[1].split("## 3.")[0]
        for k, r in summ["primary"].items():
            line = next((ln for ln in sec.splitlines() if ln.startswith(f"| {k} |")), "")
            nums = [float(x) for x in re.findall(r"[+-]?\d+\.\d+(?:e[+-]?\d+)?", line.replace("−", "-"))]
            check(f"report primary row {k}", len(nums) >= 4 and abs(nums[0] - r["mean_diff"]) <= 0.0005 and abs(nums[1] - r["ci95"][0]) <= 0.0005
                  and abs(nums[2] - r["ci95"][1]) <= 0.0005 and f"{r['b_gt_a']}/10" in line and r["label"] in line, line)
    else:
        check("PHASE5_REPORT.md exists (report checks skipped)", False, "not written yet")
    failed = [r for r in results if not r["ok"]]
    (P5 / "p5_audit.json").write_text(json.dumps({"n_checks": len(results), "n_failed": len(failed), "failed": failed}, indent=2), encoding="utf-8")
    print(f"{len(results)} checks, {len(failed)} failed")
    for r in failed:
        print("FAIL", r["check"], r["detail"])
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
