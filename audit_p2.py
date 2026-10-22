"""Independent audit of Phase 2 against PHASE2_PREREGISTRATION.md and the raw run files.

Recomputes everything from experiments/robust/p2_*_seed*/metrics.json (+ reeval.json) without
using summarize_robust.py code, and compares with summary_prefix-p2__ref-p2_full.json.
Writes experiments/robust/p2_audit.json; exits 1 if any check fails.

Usage:
    .venv/Scripts/python.exe audit_p2.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

R = Path(__file__).resolve().parent / "experiments" / "robust"
CONDITIONS = ("full", "no_o2", "o2_copy", "no_opt")
SEEDS = list(range(1, 11))
FROZEN_SHA = "c7d812e3fd8dcf2c9bd8c7f4d3faeb1bee0c97fc256f83a76ee2194e15a6436c"
OFFICIAL_TEST = {"brasilia", "montpellier", "norcia", "rio", "saclay_w", "valencia", "dubai", "lasvegas", "milano", "chongqing"}
VAL = {"aguasclaras", "beihai", "bercy", "paris"}
ARMS = {"p2_full": "full=1", "p2_dropout": "full=0.5,no_o2=0.25,no_opt=0.25"}
PREREG_ARGS = {"protocol": "holdout", "steps": 2500, "eval_every": 25, "batch_size": 64, "crop_size": 32, "lr": 5e-05,
               "weight_decay": 0.0005, "loss": "jaccard", "no_augment": False, "uniform_crop": False}
TOL = 1e-12

checks: list[dict] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append({"check": name, "pass": bool(ok), "detail": detail})


def from_counts(c: dict) -> dict:
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0, "iou": tp / (tp + fp + fn) if tp else 0.0}


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def main() -> None:
    runs = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for s in SEEDS:
            f = R / f"{arm}_seed{s}" / "metrics.json"
            check(f"{arm} seed {s} metrics.json exists", f.exists())
            if f.exists():
                runs[arm][s] = json.loads(f.read_text(encoding="utf-8"))
    if not all(len(runs[a]) == 10 for a in ARMS):
        return
    first = runs["p2_full"][1]
    for arm, conds in ARMS.items():
        for s, d in runs[arm].items():
            tag = f"{arm} seed {s}"
            a = d["args"]
            check(f"{tag}: args match pre-registration",
                  a["name"] == arm and a["seed"] == s and a["train_conditions"] == conds
                  and all(a[k] == v for k, v in PREREG_ARGS.items()), json.dumps(a))
            check(f"{tag}: train_robust.py sha256 = frozen", d["source_sha256"]["train_robust.py"] == FROZEN_SHA)
            check(f"{tag}: source hashes equal across runs", d["source_sha256"] == first["source_sha256"])
            check(f"{tag}: snapshot present", (R / f"{arm}_seed{s}" / "train_robust.py.snapshot").exists())
            tr, va, te = set(d["train_cities"]), set(d["val_cities"]), set(d["test_cities"])
            check(f"{tag}: split", te == OFFICIAL_TEST and va == VAL and len(tr) == 10 and not (tr & va) and not (tr & te) and not (va & te)
                  and tr == set(first["train_cities"]))
            # checkpoint rule: first step with max val_score; score = mean val F1 over trained conditions
            trained = [c.split("=")[0] for c in conds.split(",")]
            hist = d["history"]
            recomputed = [sum(h[f"val_f1_{c}"] for c in trained) / len(trained) for h in hist]
            check(f"{tag}: val_score = mean val F1 over trained conditions {trained}",
                  all(abs(x - h["val_score"]) < 1e-9 for x, h in zip(recomputed, hist)))
            best = hist[int(np.argmax([h["val_score"] for h in hist]))]
            check(f"{tag}: selected step = validation argmax", best["step"] == d["selected_step"]
                  and abs(best["val_score"] - d["selection_score"]) < 1e-12, f"{d['selected_step']} vs {best['step']}")
            check(f"{tag}: history covers steps 25..2500", [h["step"] for h in hist] == list(range(25, 2501, 25)))
            for c in CONDITIONS:
                t = d["test"].get(c)
                check(f"{tag} {c}: test entry present", t is not None)
                if t is None:
                    continue
                for at in ("at_selected", "at_0.5"):
                    rec = from_counts(t[at])
                    check(f"{tag} {c} {at}: metrics consistent with counts",
                          all(abs(rec[m] - t[at][m]) < 1e-9 for m in rec)
                          and sum(t[at][k] for k in ("tp", "fp", "fn", "tn")) == sum(first["test"]["full"]["at_0.5"][k] for k in ("tp", "fp", "fn", "tn")))
            rv = R / f"{arm}_seed{s}" / "reeval.json"
            ok = rv.exists() and all(v if isinstance(v, bool) else v["exact"] for v in json.loads(rv.read_text(encoding="utf-8"))["reproduction"].values())
            check(f"{tag}: reeval reproduced exactly (counts at τ and 0.5; τ = validation argmax)", ok)

    # summary file: means, sample SD, seed alignment
    summ = json.loads((R / "summary_prefix-p2__ref-p2_full.json").read_text(encoding="utf-8"))
    recomputed = {}
    for arm in ARMS:
        check(f"summary {arm}: seeds 1-10", summ[arm]["seeds"] == SEEDS)
        for c in CONDITIONS:
            for at in ("at_selected", "at_0.5"):
                for m in ("f1", "precision", "recall", "iou", "tp", "fp", "fn", "tn"):
                    vals = [runs[arm][s]["test"][c][at][m] for s in SEEDS]
                    st = summ[arm]["conditions"][c][at][m]
                    mean, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
                    recomputed[(arm, c, at, m)] = (mean, sd)
                    check(f"summary {arm} {c} {at} {m}: values/mean/sample SD",
                          summ[arm]["conditions"][c]["seeds"] == SEEDS and st["values"] == vals and st["n"] == 10
                          and math.isclose(st["mean"], mean, rel_tol=0, abs_tol=1e-9) and math.isclose(st["sd"], sd, rel_tol=0, abs_tol=1e-9))
    # paired analysis: B - A, matching seeds, t-test, Wilcoxon, Holm over 4 conditions
    rows = {r["condition"]: r for r in summ["p2_dropout"]["paired_vs_p2_full"]}
    check("paired: all 4 conditions present", set(rows) == set(CONDITIONS))
    paired = {}
    for at in ("at_selected", "at_0.5"):
        tp = {}
        for c in CONDITIONS:
            a = [runs["p2_full"][s]["test"][c][at]["f1"] for s in SEEDS]
            b = [runs["p2_dropout"][s]["test"][c][at]["f1"] for s in SEEDS]
            d = [y - x for x, y in zip(a, b)]
            tp[c] = {"diffs": d, "mean_diff": float(np.mean(d)), "sd_diff": float(np.std(d, ddof=1)),
                     "t_p": float(stats.ttest_rel(b, a).pvalue), "wilcoxon_p": float(stats.wilcoxon(b, a).pvalue),
                     "b_gt_a": sum(x > 0 for x in d),
                     "dz": float(np.mean(d) / np.std(d, ddof=1)),
                     "ci95": [float(np.mean(d) + q * np.std(d, ddof=1) / math.sqrt(10)) for q in stats.t.ppf([0.025, 0.975], 9)]}
        for c, h in zip(CONDITIONS, holm([tp[c]["t_p"] for c in CONDITIONS])):
            tp[c]["holm_p"] = h
        for c, h in zip(CONDITIONS, holm([tp[c]["wilcoxon_p"] for c in CONDITIONS])):
            tp[c]["holm_wilcoxon_p"] = h
        paired[at] = tp
    for c in CONDITIONS:
        r, mine = rows[c], paired["at_selected"][c]
        check(f"paired {c}: seeds matched 1-10", r["seeds"] == SEEDS and r["n"] == 10)
        check(f"paired {c}: diffs, t p, Wilcoxon p, Holm p reproduced",
              np.allclose(r["diffs"], mine["diffs"], atol=1e-12) and math.isclose(r["t_p"], mine["t_p"], abs_tol=1e-12)
              and math.isclose(r["wilcoxon_p"], mine["wilcoxon_p"], abs_tol=1e-12) and math.isclose(r["holm_p"], mine["holm_p"], abs_tol=1e-12))

    out = {"n_checks": len(checks), "n_failed": sum(not c["pass"] for c in checks),
           "failed": [c for c in checks if not c["pass"]], "paired": paired, "checks": checks}
    (R / "p2_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    # self-check of the Holm implementation on a textbook example
    assert np.allclose(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    main()
    failed = [c for c in checks if not c["pass"]]
    print(f"{len(checks)} checks, {len(failed)} failed")
    for c in failed[:20]:
        print("FAIL", c["check"], c["detail"][:200])
    sys.exit(1 if failed else 0)
