"""Stage 4B analysis (pre-registered in STAGE4B_PREREGISTRATION.md): model B with and without the historical image.

Reads experiments/robust/stage4b/runs (eval_stage4b.py) plus stored Step 8 counts for the SAR-only U-Net and B's
historical-image ages. Paired over the 10 training seeds, counts pooled over cities per seed, as every registered step.
Every contrast is "present - absent" (historical image in the pre-change optical slot minus both optical slots zero).

    primary            F1, 10 test cities, both conditions at B's frozen clean threshold (tau_clean, §5's threshold)
    secondary          (Holm over 3) F1 at B's frozen no_o2 threshold; precision and recall at tau_clean
    rq5_far            (Holm over the 7 registered RQ5 classes) false-alarm rate per WorldCover class, 9 primary cities,
                       both conditions at B's frozen no_o2 threshold (the registered RQ5 operating point)
    rq5_recall         as rq5_far for recall (own Holm family)
    descriptive        decomposition at tau_clean: (present - SAR-only) = (present - absent) + (absent - SAR-only);
                       FAR per class at tau_clean; each Step 8 image age - absent (9 primary cities, no_o2 threshold)
    city_level         exploratory: CITY_METHOD below

Outputs: stage4b_summary.json and STAGE4B_REPORT.md in this folder. Refuses to overwrite.

Usage (from the repository root):
    .venv/Scripts/python.exe experiments/robust/stage4b/analyze_stage4b.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analyze_phase3 import holm, paired  # noqa: E402
from analyze_phase7 import WC_NAMES, rates  # noqa: E402

R = ROOT / "experiments" / "robust"
HERE = Path(__file__).resolve().parent
SEEDS = range(1, 11)
RQ5_CLASSES = (10, 20, 30, 40, 50, 60, 80)  # the 7 classes tested in Step 8 (>= 1% of primary-city pixels)
GAPS = ("gap01", "gap03", "gap06", "gap12", "gap18")


def load():
    runs = {}
    for s in SEEDS:
        meta = json.loads((HERE / "runs" / f"p2_dropout_seed{s}.json").read_text(encoding="utf-8"))
        assert all(meta["anchors"].values()), s
        runs[s] = (meta, np.load(HERE / "runs" / f"p2_dropout_seed{s}.npz"))
    return runs


def family(pairs):
    """pairs: name -> (absent values, present values); Holm over the family on the t-test p-values."""
    rows = {k: paired(a, b) for k, (a, b) in pairs.items()}
    for k, h in zip(rows, holm([r["t_p"] for r in rows.values()])):
        rows[k]["holm_p"] = h
    for k, h in zip(rows, holm([r["wilcoxon_p"] for r in rows.values()])):
        rows[k]["holm_wilcoxon_p"] = h
    for r in rows.values():
        r["label"] = ("higher with the image" if r["holm_p"] < 0.05 and r["mean_diff"] > 0 else
                      "lower with the image" if r["holm_p"] < 0.05 and r["mean_diff"] < 0 else "no detectable difference")
    return rows


def main() -> None:
    if (HERE / "stage4b_summary.json").exists():
        raise SystemExit("stage4b_summary.json exists; refusing to overwrite")
    runs = load()
    s8 = json.loads((R / "phase8" / "summary.json").read_text(encoding="utf-8"))
    primary, all10 = s8["primary_cities"], runs[1][0]["test_cities"]
    p8 = {(a, s): (json.loads((R / "phase8" / "runs" / f"{a}_seed{s}.json").read_text(encoding="utf-8")),
                   np.load(R / "phase8" / "runs" / f"{a}_seed{s}.npz")) for a in ("p2_dropout", "p6_unet_sar") for s in SEEDS}

    def k_of(s, rule):  # frozen thresholds from the checkpoint (tau_clean = the full-input threshold, as Phase 3)
        return {"clean": runs[s][0]["thresholds"]["full"], "no_o2": runs[s][0]["thresholds"]["no_o2"], "0.5": 50}[rule]

    def met(cond, rule, metric="f1", cities=None, cls=None):
        cities = cities or all10
        sfx = f"__wc{cls}" if cls is not None else ""
        return [rates(*(float(x) for x in sum(runs[s][1][f"{c}__{cond}{sfx}"][k_of(s, rule)] for c in cities)))[metric]
                for s in SEEDS]

    sar = [rates(*(float(x) for x in sum(p8["p6_unet_sar", s][1][f"{c}__oscd"][p8["p6_unet_sar", s][0]["thresholds"]["full"]]
                                         for c in all10)))["f1"] for s in SEEDS]
    # self-checks against audited values: present at no_o2 = Step 8's B@oscd (9 primary cities); SAR-only = Step 8
    assert np.allclose(met("present", "no_o2", cities=primary), s8["arms"]["p2_dropout"]["oscd"]["f1"]["values"])
    assert np.allclose([rates(*(float(x) for x in sum(p8["p6_unet_sar", s][1][f"{c}__oscd"][p8["p6_unet_sar", s][0]["thresholds"]["full"]]
                                                        for c in primary)))["f1"] for s in SEEDS],
                       s8["arms"]["p6_unet_sar"]["oscd"]["f1"]["values"])

    out = {"test_cities": all10, "primary_cities": primary}
    out["primary"] = paired(met("absent", "clean"), met("present", "clean"))
    out["primary"]["label"] = ("higher with the image" if out["primary"]["t_p"] < 0.05 and out["primary"]["mean_diff"] > 0 else
                               "lower with the image" if out["primary"]["t_p"] < 0.05 and out["primary"]["mean_diff"] < 0
                               else "no detectable difference")
    out["secondary"] = family({"F1 at the no_o2 threshold": (met("absent", "no_o2"), met("present", "no_o2")),
                               "precision at tau_clean": (met("absent", "clean", "precision"), met("present", "clean", "precision")),
                               "recall at tau_clean": (met("absent", "clean", "recall"), met("present", "clean", "recall"))})
    out["rq5_far"] = family({WC_NAMES[c]: (met("absent", "no_o2", "false_alarm_rate", primary, c),
                                           met("present", "no_o2", "false_alarm_rate", primary, c)) for c in RQ5_CLASSES})
    out["rq5_recall"] = family({WC_NAMES[c]: (met("absent", "no_o2", "recall", primary, c),
                                              met("present", "no_o2", "recall", primary, c)) for c in RQ5_CLASSES})
    out["descriptive"] = {
        "decomposition_tau_clean": {"present - SAR-only": paired(sar, met("present", "clean")),
                                    "present - absent": out["primary"],
                                    "absent - SAR-only": paired(sar, met("absent", "clean"))},
        "far_by_class_tau_clean": {WC_NAMES[c]: paired(met("absent", "clean", "false_alarm_rate", primary, c),
                                                       met("present", "clean", "false_alarm_rate", primary, c)) for c in RQ5_CLASSES},
        "image_age_minus_absent": {g: paired(met("absent", "no_o2", cities=primary),
                                             [rates(*(float(x) for x in sum(p8["p2_dropout", s][1][f"{c}__{g}"][k_of(s, "no_o2")]
                                                                            for c in primary)))["f1"] for s in SEEDS]) for g in GAPS}}
    out["city_level"] = city_level(runs, all10)
    (HERE / "stage4b_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (HERE / "STAGE4B_REPORT.md").write_text(report(out), encoding="utf-8")
    print("wrote stage4b_summary.json and STAGE4B_REPORT.md")


def f1c(c):
    tp, fp, fn = float(c[0]), float(c[1]), float(c[2])
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


def city_level(runs, cities, boot=10000):
    """Exploratory, primary contrast only (tau_clean, 10 cities). Method as corrected after the separate (same-assistant) check of
    Stage 4 E2: SEs of the same pixel-pooled estimate under seed and city resampling, a paired two-stage percentile
    bootstrap, a jackknife-t interval, and per-city deltas (unweighted) with each city's share of change pixels."""
    rng = np.random.default_rng(20261103)
    k = {s: runs[s][0]["thresholds"]["full"] for s in SEEDS}
    X = np.array([[runs[s][1][f"{c}__present"][k[s]] for c in cities] for s in SEEDS], dtype=float)
    Y = np.array([[runs[s][1][f"{c}__absent"][k[s]] for c in cities] for s in SEEDS], dtype=float)
    pooled = lambda C, ci: np.array([f1c(C[i, ci].sum(0)) for i in range(len(C))])
    ns, nc = X.shape[:2]
    allc = np.arange(nc)
    d = pooled(X, allc) - pooled(Y, allc)
    two = np.empty(boot)
    for b in range(boot):  # same city and seed indices for both conditions: the pairing is kept
        ci, si = rng.integers(0, nc, nc), rng.integers(0, ns, ns)
        two[b] = (pooled(X, ci) - pooled(Y, ci))[si].mean()
    jk = np.array([(pooled(X, np.delete(allc, j)) - pooled(Y, np.delete(allc, j))).mean() for j in range(nc)])
    se_jk = float(np.sqrt((nc - 1) / nc * ((jk - jk.mean()) ** 2).sum()))
    se_seed = float(d.std(ddof=1) / np.sqrt(ns))
    t = stats.t.ppf(0.975, ns - 1)
    per_city = np.array([[f1c(X[i, j]) - f1c(Y[i, j]) for j in range(nc)] for i in range(ns)]).mean(0)
    pos = X[0, :, 0] + X[0, :, 2]
    return {"mean_diff": float(d.mean()), "se_seeds": se_seed, "se_cities_jackknife": se_jk,
            "two_stage_percentile_ci95": np.percentile(two, [2.5, 97.5]).tolist(),
            "jackknife_t_ci95": [float(d.mean() - t * np.hypot(se_jk, se_seed)), float(d.mean() + t * np.hypot(se_jk, se_seed))],
            "loco_range": [float(jk.min()), float(jk.max())], "loco_mean_diff": dict(zip(cities, jk.tolist())),
            "per_city_mean_diff": dict(zip(cities, per_city.tolist())), "cities_same_sign": int((np.sign(per_city) == np.sign(d.mean())).sum()),
            "change_pixel_share": dict(zip(cities, (pos / pos.sum()).tolist()))}


def report(out):
    f = lambda x: f"{x:+.3f}"
    ci = lambda c: f"[{f(c[0])}, {f(c[1])}]"
    row = lambda name, r: (f"| {name} | {f(r['mean_diff'])} | {ci(r['ci95'])} | {r['dz']:.2f} | {r['b_gt_a']}/10 | "
                           f"{r['t_p']:.3g} | {r.get('holm_p', r['t_p']):.3g} | {r['wilcoxon_p']:.3g} | {r.get('label', '')} |")
    H = ["| contrast | Δ (present − absent) | 95% CI | d_z | Δ > 0 | p | Holm p | Wilcoxon p | label |", "|---|---|---|---|---|---|---|---|---|"]
    p, c = out["primary"], out["city_level"]
    L = ["# Stage 4B: model B with and without the historical optical image", "",
         "Pre-registered in `STAGE4B_PREREGISTRATION.md`. Generated by `analyze_stage4b.py`; numbers in `stage4b_summary.json`.", "",
         "## Primary: F1, 10 test cities, both conditions at B's frozen clean threshold", "", *H, row("F1 at tau_clean", p), "",
         "## Secondary (Holm over 3)", "", *H, *[row(k, r) for k, r in out["secondary"].items()], "",
         "## RQ5: false-alarm rate by WorldCover class, 9 primary cities, B's no_o2 threshold (Holm over 7)", "", *H,
         *[row(k, r) for k, r in out["rq5_far"].items()], "", "## RQ5: recall by class (Holm over 7)", "", *H,
         *[row(k, r) for k, r in out["rq5_recall"].items()], "", "## Descriptive (no tests)", "",
         "Decomposition at tau_clean (10 cities):", "", *H, *[row(k, r) for k, r in out["descriptive"]["decomposition_tau_clean"].items()], "",
         "False-alarm rate by class at tau_clean (9 primary cities):", "", *H,
         *[row(k, r) for k, r in out["descriptive"]["far_by_class_tau_clean"].items()], "",
         "Each Step 8 image age minus absent (F1, 9 primary cities, no_o2 threshold):", "", *H,
         *[row(k, r) for k, r in out["descriptive"]["image_age_minus_absent"].items()], "",
         "## City level (exploratory; primary contrast)", "",
         f"Δ {f(c['mean_diff'])}; SE over seeds {c['se_seeds']:.3f}, over cities (jackknife) {c['se_cities_jackknife']:.3f}; "
         f"two-stage percentile 95% CI {ci(c['two_stage_percentile_ci95'])}; jackknife-t 95% CI {ci(c['jackknife_t_ci95'])}; "
         f"one city left out {f(c['loco_range'][0])} to {f(c['loco_range'][1])}; cities with the same sign "
         f"{c['cities_same_sign']}/{len(c['per_city_mean_diff'])} (unweighted).", "",
         "| city | Δ | pooled Δ without it | share of change pixels |", "|---|---|---|---|",
         *[f"| {k} | {f(v)} | {f(c['loco_mean_diff'][k])} | {100 * c['change_pixel_share'][k]:.0f}% |" for k, v in c["per_city_mean_diff"].items()]]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
