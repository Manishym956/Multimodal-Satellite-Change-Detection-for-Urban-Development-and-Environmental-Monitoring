"""Independent audit of Step 4A-bis: masks -> manifest, raw counts -> summary.json -> PHASE4ABIS_REPORT.md.

Recomputes from phase4abis/runs/*.npz with numpy/scipy only (not analyze_phase4abis.py or analyze_phase3.py code). Checks:
  hashes: every file hash in PHASE4ABIS_PREREGISTRATION.md equals the file on disk
  masks:  file hash = manifest; P built independently from the oracle M (scipy erosion/dilation with a disk; for miss,
          P subset of M with exactly round(Q|M|) pixels removed); manifest error fractions; test-city mean error stats
  runs:   20 present, anchors true, one source-hash set, eval_phase4abis.py hash = registered, frozen threshold =
          Phase 2 clean-validation threshold; reference conditions equal the stored Step 4A counts
  stats:  per-seed F1 (frozen clean threshold) for every arm x condition, mean, SD; retained gain; every Holm family
          (diffs, CI, d_z, t p, Holm, Wilcoxon, Holm-Wilcoxon); region FAR / recall
  report: every number in the PHASE4ABIS_REPORT.md primary table equals the recomputed value (3 decimals)
Writes experiments/robust/phase4abis/p4abis_audit.json; exits 1 on any failure.

Usage:
    .venv/Scripts/python.exe audit_p4abis.py
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
R = ROOT / "experiments" / "robust"
P3, P4A, OUT = R / "phase3", R / "phase4a", R / "phase4abis"
LEVELS = ("c10", "c30", "c60")
VARIANTS = ("erode2", "erode4", "erode8", "dilate2", "dilate4", "dilate8", "miss10", "miss25", "miss50")
ARMS = ("p2_full", "p2_dropout")
SEEDS = range(1, 11)
checks = []


def check(name, ok, detail=""):
    checks.append({"check": name, "pass": bool(ok), "detail": str(detail)[:300]})


def rate(counts, k, m):
    tp, fp, fn, tn = (int(v) for v in counts[k])
    num, den = {"f1": (2 * tp, 2 * tp + fp + fn), "precision": (tp, tp + fp), "recall": (tp, tp + fn),
                "false_alarm_rate": (fp, fp + tn), "iou": (tp, tp + fp + fn)}[m]
    return num / den if den else 0.0


def holm(p):
    adj, run = [0.0] * len(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = run
    return adj


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def close(a, b, tol=1e-12):
    return math.isclose(a, b, abs_tol=tol) or (math.isnan(a) and math.isnan(b))


def main():
    prereg = (R / "PHASE4ABIS_PREREGISTRATION.md").read_text(encoding="utf-8")
    for path, sha in re.findall(r"\| `([^`*]+?)`[^|]*\| `([0-9a-f]{64})` \|", prereg):
        f = ROOT / path if (ROOT / path).exists() else R / path
        check(f"registered hash {path}", f.exists() and hashlib.sha256(f.read_bytes()).hexdigest() == sha, path)
    registered_eval = re.search(r"`eval_phase4abis.py` \| `([0-9a-f]{64})`", prereg).group(1)

    manifest = json.loads((OUT / "mask_manifest.json").read_text(encoding="utf-8"))
    check("manifest: 14 cities x 3 levels x 9 variants", len(manifest) == 378 and len({e["city"] for e in manifest}) == 14)
    for e in manifest:
        path = OUT / "masks" / e["file"]
        oracle_path = P3 / "masks" / f"{e['city']}_{e['level']}.npy"
        m, p = np.load(oracle_path), np.load(path)
        ok = hashlib.sha256(path.read_bytes()).hexdigest() == e["sha256"] and hashlib.sha256(oracle_path.read_bytes()).hexdigest() == e["oracle_sha256"]
        v = e["variant"]
        if v.startswith("erode"):
            ok &= np.array_equal(p, ndimage.binary_erosion(m, disk(int(v[5:])), border_value=1))
        elif v.startswith("dilate"):
            ok &= np.array_equal(p, ndimage.binary_dilation(m, disk(int(v[6:]))))
        else:
            ok &= not (p & ~m).any() and int((m & ~p).sum()) == int(np.round(int(v[4:]) / 100 * m.sum()))
        missed, over = (m & ~p).sum() / m.sum(), (p & ~m).sum() / (~m).sum()
        ok &= close(missed, e["missed_cloud_frac"], 1e-9) and close(over, e["overmasked_clear_frac"], 1e-9)
        check(f"mask {e['file']}: hash, construction, error fractions", ok)

    runs = {}
    for arm in ARMS:
        for s in SEEDS:
            doc = json.loads((OUT / "runs" / f"{arm}_seed{s}.json").read_text(encoding="utf-8"))
            npz = np.load(OUT / "runs" / f"{arm}_seed{s}.npz")
            runs[(arm, s)] = (doc, npz)
            p2 = json.loads((R / f"{arm}_seed{s}" / "metrics.json").read_text(encoding="utf-8"))["test"]
            p4a = np.load(P4A / "runs" / f"{arm}_seed{s}.npz")
            check(f"{arm} seed {s}: anchors true", all(doc["anchors"].values()))
            check(f"{arm} seed {s}: eval_phase4abis.py = registered hash", doc["source_sha256"]["eval_phase4abis.py"] == registered_eval)
            check(f"{arm} seed {s}: frozen threshold = Phase 2 clean-validation threshold", doc["tau_clean"] == p2["full"]["selected_threshold"])
            refs = ("clean", *LEVELS, "z10", "z30", "z60")
            check(f"{arm} seed {s}: references equal stored Step 4A counts (val and test)",
                  all(np.array_equal(npz[f"{t}__{c}"], p4a[f"{t}__{c}"]) for c in refs for t in ("val", "test")))
    check("one source-hash set across runs", len({json.dumps(d["source_sha256"], sort_keys=True) for d, _ in runs.values()}) == 1)

    summ = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    tau = {k: round(d["tau_clean"] * 100) for k, (d, _) in runs.items()}
    conds = ["clean", *LEVELS, "z10", "z30", "z60", *(f"{lv}__{v}" for lv in LEVELS for v in VARIANTS)]
    f1 = {}
    for arm in ARMS:
        for c in conds:
            for m in ("f1", "precision", "recall", "false_alarm_rate"):
                vals = [rate(runs[(arm, s)][1][f"test__{c}"], tau[(arm, s)], m) for s in SEEDS]
                st = summ["arms"][arm]["clean"][c][m]
                check(f"summary {arm} {c} {m}: per-seed values, mean, SD",
                      np.allclose(st["values"], vals, atol=1e-12) and close(st["mean"], np.mean(vals)) and close(st["sd"], np.std(vals, ddof=1)))
                if m == "f1":
                    f1[(arm, c)] = np.array(vals)
        for lv in LEVELS:
            for v in VARIANTS:
                ret = (f1[(arm, f"{lv}__{v}")] - f1[(arm, lv)]) / (f1[(arm, f"z{lv[1:]}")] - f1[(arm, lv)])
                st = summ["arms"][arm]["clean"][f"{lv}__{v}"]["retained_gain"]
                check(f"retained gain {arm} {lv}__{v}", np.allclose(st["values"], ret, atol=1e-12)
                      and st["most_retained"] == bool(ret.mean() >= 0.75))
    p4a_summ = json.loads((P4A / "summary.json").read_text(encoding="utf-8"))
    for arm in ARMS:
        for c in ("clean", *LEVELS, "z10", "z30", "z60"):
            check(f"{arm} {c}: per-seed F1 equals Step 4A summary", np.allclose(p4a_summ["arms"][arm]["clean"][c]["f1"]["values"], f1[(arm, c)], atol=1e-12))

    def family(pairs):
        rows = {}
        for key, (a, b) in pairs.items():
            d = b - a
            half = stats.t.ppf(0.975, 9) * d.std(ddof=1) / math.sqrt(10)
            rows[key] = {"mean_diff": d.mean(), "ci95": (d.mean() - half, d.mean() + half), "dz": d.mean() / d.std(ddof=1),
                         "b_gt_a": int((d > 0).sum()), "t_p": stats.ttest_rel(b, a).pvalue, "wilcoxon_p": stats.wilcoxon(b, a).pvalue}
        for key, h in zip(rows, holm([r["t_p"] for r in rows.values()])):
            rows[key]["holm_p"] = h
        for key, h in zip(rows, holm([r["wilcoxon_p"] for r in rows.values()])):
            rows[key]["holm_wilcoxon_p"] = h
        return rows

    expected = {"primary_B_c30_imperfect_minus_oracle": family({v: (f1[("p2_dropout", "z30")], f1[("p2_dropout", f"c30__{v}")]) for v in VARIANTS})}
    for arm in ARMS:
        for lv in LEVELS:
            if (arm, lv) != ("p2_dropout", "c30"):
                expected[f"secondary_{arm}_{lv}_imperfect_minus_oracle"] = family(
                    {v: (f1[(arm, f"z{lv[1:]}")], f1[(arm, f"{lv}__{v}")]) for v in VARIANTS})
    expected["secondary_B_minus_A_c30"] = family({v: (f1[("p2_full", f"c30__{v}")], f1[("p2_dropout", f"c30__{v}")]) for v in VARIANTS})
    check("summary has exactly the registered families (1 primary + 6 secondary)", set(summ["families"]) == set(expected), sorted(summ["families"]))
    for name, rows in expected.items():
        for key, r in rows.items():
            s = summ["families"].get(name, {}).get(key, {})
            ok = bool(s) and all(close(s[k], r[k], 1e-9) for k in ("mean_diff", "dz", "t_p", "holm_p", "wilcoxon_p", "holm_wilcoxon_p"))
            ok = ok and np.allclose(s["ci95"], r["ci95"], atol=1e-12) and s["b_gt_a"] == r["b_gt_a"]
            check(f"family {name} {key}", ok)

    for arm in ARMS:
        for lv in LEVELS:
            for v in VARIANTS:
                c = f"{lv}__{v}"
                for reg in ("missed", "over"):
                    for src, table in (("clean", f"{reg}_clean"), ("input", reg)):
                        for m in ("false_alarm_rate", "recall"):
                            vals = [rate(runs[(arm, s)][1][f"{table}__{c}"], tau[(arm, s)], m) for s in SEEDS]
                            st = summ["regions"][arm][c][f"{reg}_{src}_{m}"]
                            check(f"region {arm} {c} {reg}_{src}_{m}", np.allclose(st["values"], vals, atol=1e-12))

    test_rows = [e for e in manifest if e["split"] == "test"]
    for lv in LEVELS:
        for v in VARIANTS:
            rows = [e for e in test_rows if e["level"] == lv and e["variant"] == v]
            es = summ["error_stats"][f"{lv}__{v}"]
            check(f"error stats {lv}__{v} (10 test cities)", len(rows) == 10
                  and close(es["missed_cloud_frac"], np.mean([e["missed_cloud_frac"] for e in rows]))
                  and close(es["overmasked_clear_frac"], np.mean([e["overmasked_clear_frac"] for e in rows])))

    report_path = R / "PHASE4ABIS_REPORT.md"
    if report_path.exists():
        section = report_path.read_text(encoding="utf-8").split("## 2.")[1].split("## 3.")[0]
        prim = expected["primary_B_c30_imperfect_minus_oracle"]
        for v in VARIANTS:
            line = next((l for l in section.splitlines() if l.startswith(f"| {v} |")), "")
            nums = [float(x) for x in re.findall(r"[+-]?\d+\.\d+(?:e[+-]?\d+)?", line.replace("−", "-"))]
            r, b = prim[v], f1[("p2_dropout", f"c30__{v}")]
            es = [e for e in manifest if e["split"] == "test" and e["level"] == "c30" and e["variant"] == v]
            expect = [np.mean([e["missed_cloud_frac"] for e in es]), np.mean([e["overmasked_clear_frac"] for e in es]),
                      b.mean(), b.std(ddof=1), r["mean_diff"], r["ci95"][0], r["ci95"][1], r["dz"]]
            tols = [0.0005] * 7 + [0.005]
            check(f"report primary row {v}", len(nums) >= 9 and all(abs(x - y) <= t + 1e-9 for x, y, t in zip(nums, expect, tols))
                  and math.isclose(nums[8], r["holm_p"], rel_tol=0.06) and f"{r['b_gt_a']}/10" in line, line)
    else:
        check("PHASE4ABIS_REPORT.md exists (report checks skipped)", False, "not written yet")

    out = {"n_checks": len(checks), "n_failed": sum(not c["pass"] for c in checks), "failed": [c for c in checks if not c["pass"]], "checks": checks}
    (OUT / "p4abis_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    assert np.allclose(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    main()
    failed = [c for c in checks if not c["pass"]]
    print(f"{len(checks)} checks, {len(failed)} failed")
    for c in failed:
        print("FAIL", c["check"], c["detail"])
    sys.exit(1 if failed else 0)
