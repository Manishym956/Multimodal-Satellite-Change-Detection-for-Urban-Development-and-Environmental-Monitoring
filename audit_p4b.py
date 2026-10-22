"""Independent audit of Step 4B (written before any p4b_spatial run): runs -> pre-registration, raw counts -> summary.json.

Recomputes with numpy/scipy only (not analyze_phase4b.py / analyze_phase3.py code). Checks:
  training: 10 C runs, arguments = registered (mixture, steps, eval cadence, optimiser, batch, crop, seed), source hashes =
            registered, city split, selected step = first argmax of the validation history, selection score = mean of the
            full / no_o2 / no_opt validation F1, checkpoint step and thresholds = metrics.json
  eval:     20 eval files present (C) and reused B files, anchors true, one source-hash set, eval_phase4b.py = registered
  stats:    per-seed F1 at the frozen clean threshold; primary family (Δ, CI, d_z, t p, Holm, Wilcoxon, Holm-W, labels,
            clean-regression flag); both secondary families; descriptive metrics and inside/outside means
  report:   if PHASE4B_REPORT.md exists, every number in its primary table equals the recomputation
Writes experiments/robust/phase4b/p4b_audit.json; exits 1 on any failure.

Usage:
    .venv/Scripts/python.exe audit_p4b.py
"""

import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parent
R = ROOT / "experiments" / "robust"
PRE = R / "PHASE4B_PREREGISTRATION.md"
SEEDS = range(1, 11)
PRIMARY = ("clean", "z10", "z30", "z60", "c30__erode4", "c30__dilate4", "c30__miss25")
LEVELS = ("c10", "c30", "c60")
VARIANTS = ("erode2", "erode4", "erode8", "dilate2", "dilate4", "dilate8", "miss10", "miss25", "miss50")
CONDS = ["clean", *LEVELS, *(f"z{lv[1:]}" for lv in LEVELS), *(f"{lv}__{v}" for lv in LEVELS for v in VARIANTS)]
P2 = ("full", "no_o2", "no_opt", "o2_copy")
ARGS = {"steps": 2500, "eval_every": 25, "batch_size": 64, "crop_size": 32, "lr": 5e-5, "weight_decay": 5e-4,
        "train_conditions": "full=0.5,spatial=0.25,no_o2=0.125,no_opt=0.125", "protocol": "holdout", "loss": "jaccard",
        "no_augment": False, "uniform_crop": False}
checks = []


def check(name, ok, detail=""):
    checks.append({"check": name, "pass": bool(ok), "detail": str(detail)[:300]})


def rate(counts, k, m):
    tp, fp, fn, tn = (int(v) for v in counts[k])
    num, den = {"f1": (2 * tp, 2 * tp + fp + fn), "precision": (tp, tp + fp), "recall": (tp, tp + fn),
                "iou": (tp, tp + fp + fn), "false_alarm_rate": (fp, fp + tn)}[m]
    return num / den if den else 0.0


def holm(p):
    adj, run = [0.0] * len(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = run
    return adj


def close(a, b, tol=1e-9):
    return (a is None and b is None) or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)) \
        or math.isclose(a, b, rel_tol=tol, abs_tol=1e-12)


def family(pairs):
    rows = {}
    for key, (b, c) in pairs.items():
        d = np.asarray(c) - np.asarray(b)
        sd = d.std(ddof=1)
        half = stats.t.ppf(0.975, 9) * sd / math.sqrt(10)
        rows[key] = {"mean_diff": float(d.mean()), "ci95": (d.mean() - half, d.mean() + half), "dz": float(d.mean() / sd) if sd else float("nan"),
                     "b_gt_a": int((d > 0).sum()), "t_p": float(stats.ttest_rel(c, b).pvalue) if sd else 1.0,
                     "wilcoxon_p": float(stats.wilcoxon(c, b).pvalue) if d.any() else 1.0}
    for key, h in zip(rows, holm([r["t_p"] for r in rows.values()])):
        rows[key]["holm_p"] = h
    for key, h in zip(rows, holm([r["wilcoxon_p"] for r in rows.values()])):
        rows[key]["holm_wilcoxon_p"] = h
    return rows


def main():
    pre = PRE.read_text(encoding="utf-8")
    reg = dict(re.findall(r"^\| `([^`]+)` \| `([0-9a-f]{64})` \|", pre, re.M))
    expect_src = {"train_spatial.py": reg["src/training/train_spatial.py"], "train_robust.py": reg["src/training/train_robust.py"],
                  "oscd_dataset.py": reg["src/datasets/oscd_dataset.py"], "networks.py": reg["repos/2/DS_UNet/utils/networks.py"],
                  "loss_functions.py": reg["repos/2/DS_UNet/utils/loss_functions.py"], "spatial_mask_bank.npy": reg["experiments/robust/phase4b/spatial_mask_bank.npy"]}
    cities = json.loads((R / "p2_dropout_seed1" / "metrics.json").read_text(encoding="utf-8"))
    met = {}
    for s in SEEDS:
        m = json.loads((R / f"p4b_spatial_seed{s}" / "metrics.json").read_text(encoding="utf-8"))
        met[s] = m
        a = m["args"]
        check(f"C seed {s}: arguments = registered", a["seed"] == s and a["name"] == "p4b_spatial" and all(a[k] == v for k, v in ARGS.items()), a)
        check(f"C seed {s}: source hashes = registered", m["source_sha256"] == expect_src, m["source_sha256"])
        check(f"C seed {s}: snapshot = registered train_spatial.py",
              hashlib.sha256((R / f"p4b_spatial_seed{s}" / "train_spatial.py.snapshot").read_bytes()).hexdigest() == expect_src["train_spatial.py"])
        check(f"C seed {s}: city split = Phase 2", all(m[k] == cities[k] for k in ("train_cities", "val_cities", "test_cities"))
              and not set(m["test_cities"]) & set(m["train_cities"] + m["val_cities"]))
        h = m["history"]
        check(f"C seed {s}: 100 validation points every 25 steps", [e["step"] for e in h] == list(range(25, 2501, 25)))
        check(f"C seed {s}: selection score = mean val F1 over full/no_o2/no_opt",
              all(math.isclose(e["val_score"], np.mean([e[f"val_f1_{c}"] for c in ("full", "no_o2", "no_opt")]), abs_tol=1e-12) for e in h)
              and m["selected_on"] == ["full", "no_o2", "no_opt"])
        best = max(h, key=lambda e: e["val_score"])  # max returns the first maximum, as the strict '>' in training
        check(f"C seed {s}: selected step = first argmax of validation score", m["selected_step"] == best["step"] and m["selection_score"] == best["val_score"])
        ck = torch.load(ROOT / "checkpoints" / "robust" / f"p4b_spatial_seed{s}.pt", map_location="cpu", weights_only=False)
        check(f"C seed {s}: checkpoint step and thresholds = metrics.json",
              ck["step"] == m["selected_step"] and all(ck["thresholds"][c] / 100 == m["test"][c]["selected_threshold"] for c in P2))

    reg_eval = reg["eval_phase4b.py"]
    runs = {}
    for arm, folder in (("p2_dropout", R / "phase4abis" / "runs"), ("p4b_spatial", R / "phase4b" / "runs")):
        for s in SEEDS:
            doc = json.loads((folder / f"{arm}_seed{s}.json").read_text(encoding="utf-8"))
            runs[(arm, s)] = (doc, np.load(folder / f"{arm}_seed{s}.npz"))
            check(f"{arm} seed {s}: eval anchors true", all(doc["anchors"].values()), doc["anchors"])
    cdocs = [runs[("p4b_spatial", s)][0] for s in SEEDS]
    check("C eval: one source-hash set; eval_phase4b.py = registered",
          len({json.dumps(d["source_sha256"], sort_keys=True) for d in cdocs}) == 1 and cdocs[0]["source_sha256"]["eval_phase4b.py"] == reg_eval)
    for s in SEEDS:
        d = runs[("p4b_spatial", s)][0]
        check(f"C seed {s}: frozen threshold = training clean threshold", d["tau_clean"] == met[s]["test"]["full"]["selected_threshold"])

    tau = {k: round(d["tau_clean"] * 100) for k, (d, _) in runs.items()}
    f1 = {(arm, c): [rate(runs[(arm, s)][1][f"test__{c}"], tau[(arm, s)], "f1") for s in SEEDS] for arm in ("p2_dropout", "p4b_spatial") for c in CONDS}
    summ = json.loads((R / "phase4b" / "summary.json").read_text(encoding="utf-8"))
    for arm in ("p2_dropout", "p4b_spatial"):
        for c in CONDS:
            st = summ["arms"][arm][c]
            check(f"summary {arm} {c}: per-seed F1, mean, SD", np.allclose(st["values"], f1[(arm, c)], atol=1e-12)
                  and close(st["mean"], float(np.mean(f1[(arm, c)]))) and close(st["sd"], float(np.std(f1[(arm, c)], ddof=1))))
    prim = family({c: (f1[("p2_dropout", c)], f1[("p4b_spatial", c)]) for c in PRIMARY})
    sec = family({c: (f1[("p2_dropout", c)], f1[("p4b_spatial", c)]) for c in CONDS if c not in PRIMARY})
    p2b = {c: [json.loads((R / f"p2_dropout_seed{s}" / "metrics.json").read_text(encoding="utf-8"))["test"][c]["at_selected"]["f1"] for s in SEEDS] for c in P2}
    p2c = {c: [met[s]["test"][c]["at_selected"]["f1"] for s in SEEDS] for c in P2}
    sec2 = family({c: (p2b[c], p2c[c]) for c in P2})
    for name, mine in (("primary", prim), ("secondary_other_conditions", sec), ("secondary_phase2_conditions", sec2)):
        for c, r in mine.items():
            s_ = summ[name][c]
            ok = all(close(s_[k], r[k]) for k in ("mean_diff", "t_p", "holm_p", "wilcoxon_p", "holm_wilcoxon_p")) and s_["b_gt_a"] == r["b_gt_a"]
            ok = ok and np.allclose(s_["ci95"], r["ci95"], atol=1e-12) and (close(s_["dz"], r["dz"]))
            check(f"family {name} {c}", ok, (s_, r))
    for c, r in prim.items():
        lab = "C better" if r["holm_p"] < 0.05 and r["mean_diff"] > 0 else "C worse" if r["holm_p"] < 0.05 and r["mean_diff"] < 0 else "no detectable difference"
        check(f"label {c}", summ["primary"][c]["label"] == lab, (summ["primary"][c]["label"], lab))
    check("clean-regression flag (mean Δ < -0.02)", summ["substantial_clean_regression"] == (prim["clean"]["mean_diff"] < -0.02))
    for arm in ("p2_dropout", "p4b_spatial"):
        for c in PRIMARY:
            for m in ("precision", "recall", "iou", "false_alarm_rate"):
                vals = [rate(runs[(arm, s)][1][f"test__{c}"], tau[(arm, s)], m) for s in SEEDS]
                check(f"metric {arm} {c} {m}", np.allclose(summ["metrics"][arm][c][m]["values"], vals, atol=1e-12))
    for arm, folder, prefix in (("p2_dropout", R / "phase4abis" / "addendum_inside_outside" / "runs", ""), ("p4b_spatial", R / "phase4b" / "runs", "io_")):
        arrs = [np.load(folder / f"{arm}_seed{s}.npz") for s in SEEDS]
        for c, e in summ["inside_outside"][arm].items():
            for part in ("inside", "outside"):
                for m in ("f1", "recall", "false_alarm_rate"):
                    vals = [rate(a[f"{prefix}{part}__{c}"], tau[(arm, s)], m) for a, s in zip(arrs, SEEDS)]
                    check(f"inside/outside {arm} {c} {part} {m}", np.allclose(e[f"{part}_{m}"]["values"], vals, atol=1e-12))

    rep = R / "PHASE4B_REPORT.md"
    if rep.exists():
        section = rep.read_text(encoding="utf-8").split("## 2.")[1].split("## 3.")[0]
        for c in PRIMARY:
            line = next((l for l in section.splitlines() if l.startswith(f"| {c} |")), "")
            nums = [float(x) for x in re.findall(r"[+-]?\d+\.\d+(?:e[+-]?\d+)?", line.replace("−", "-"))]
            r, b, cc = prim[c], np.array(f1[("p2_dropout", c)]), np.array(f1[("p4b_spatial", c)])
            expect = [b.mean(), b.std(ddof=1), cc.mean(), cc.std(ddof=1), r["mean_diff"], r["ci95"][0], r["ci95"][1]]
            check(f"report primary row {c}", len(nums) >= 9 and all(abs(x - y) <= 0.0005 + 1e-9 for x, y in zip(nums, expect))
                  and math.isclose(nums[8], r["holm_p"], rel_tol=0.06) and f"{r['b_gt_a']}/10" in line and summ["primary"][c]["label"] in line, line)
    else:
        check("PHASE4B_REPORT.md exists (report checks skipped)", False, "not written yet")
    out = {"n_checks": len(checks), "n_failed": sum(not c["pass"] for c in checks), "failed": [c for c in checks if not c["pass"]], "checks": checks}
    (R / "phase4b" / "p4b_audit.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    assert np.allclose(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    main()
    failed = [c for c in checks if not c["pass"]]
    print(f"{len(checks)} checks, {len(failed)} failed")
    for c in failed:
        print("FAIL", c["check"], c["detail"])
    sys.exit(1 if failed else 0)
