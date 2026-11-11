"""Step 10 analysis (pre-registered in STEP10_PREREGISTRATION.md): confirmatory replication on fresh seeds 11-20.

E = p10_fusion_dropout (Step 9's early-fusion modality-dropout U-Net), S = p10_unet_sar (Step 6's SAR-only U-Net), both
retrained on seeds 11-20. Test F1 at each run's validation-selected threshold for the condition (Step 6's rule).
    primary          E@no_opt - S@full (single test, alpha 0.05)
    secondary        E@no_o2 - E@no_opt (Step 9 H1, replication); E@no_opt - S@full at threshold 0.5     (Holm over 2)
    city level       (exploratory) one-city-left-out range and jackknife SE of the primary contrast, per-city sign counts
    descriptive      all metrics; the primary contrast over all 20 seeds (seeds 1-10 from Step 9, not confirmatory)
Writes summary.json and STEP10_REPORT.md next to this file; refuses to overwrite.

Usage (repository root):  .venv/Scripts/python.exe experiments/robust/step10/analyze_step10.py [--runs D --cities D --out D]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "robust" / "step9"))
from analyze_phase3 import describe, paired  # noqa: E402
from analyze_step9 import family, family_table  # noqa: E402
from src.evaluation.metrics import rates_from_counts  # noqa: E402

SEEDS = range(11, 21)
CONDS = {"p10_fusion_dropout": ("full", "no_o2", "no_opt", "o2_copy"), "p10_unet_sar": ("full",)}
NAMES = {"p10_fusion_dropout": "E early-fusion U-Net, modality dropout", "p10_unet_sar": "S SAR-only U-Net"}
SETTINGS = {"p10_fusion_dropout": {"arch": "unet_fusion", "train_conditions": "full=0.5,no_o2=0.25,no_opt=0.25"},
            "p10_unet_sar": {"arch": "unet_sar", "train_conditions": "full=1"}}
COMMON = {"sar": "hafner_vv", "steps": 2500, "eval_every": 25, "batch_size": 64, "crop_size": 32, "lr": 5e-5, "weight_decay": 5e-4}
METRICS = ("f1", "precision", "recall", "iou", "false_alarm_rate")


def f1(c) -> float:
    return rates_from_counts({"tp": int(c[0]), "fp": int(c[1]), "fn": int(c[2]), "tn": int(c[3])})["f1"] or 0.0


def load(runs: Path) -> dict:
    out = {(a, s): json.loads((runs / f"{a}_seed{s}" / "metrics.json").read_text(encoding="utf-8")) for a in CONDS for s in SEEDS}
    assert len({tuple(m["test_cities"]) for m in out.values()}) == 1 and len({tuple(m["val_cities"]) for m in out.values()}) == 1
    for (a, s), m in out.items():
        assert all(m["args"][k] == v for k, v in {**COMMON, **SETTINGS[a]}.items()) and m["args"]["seed"] == s, f"{a} {s} settings"
        for c in CONDS[a]:
            for thr in ("at_selected", "at_0.5"):
                r = m["test"][c][thr]
                assert f1((r["tp"], r["fp"], r["fn"], r["tn"])) == (r["f1"] or 0.0)
    return out


def city_level(cities_dir: Path, runs: dict) -> dict:
    """Primary contrast with one test city left out at a time, from per-city counts at the selected thresholds."""
    test = runs["p10_unet_sar", 11]["test_cities"]
    e = {s: np.load(cities_dir / f"p10_fusion_dropout_seed{s}.npz") for s in SEEDS}
    sr = {s: np.load(cities_dir / f"p10_unet_sar_seed{s}.npz") for s in SEEDS}
    k = lambda a, s, c: round(runs[a, s]["test"][c]["selected_threshold"] * 100)
    ce = lambda s, city: e[s][f"{city}__no_opt"][k("p10_fusion_dropout", s, "no_opt")]
    cs = lambda s, city: sr[s][f"{city}__full"][k("p10_unet_sar", s, "full")]
    delta = lambda keep: float(np.mean([f1(sum(ce(s, c) for c in keep)) - f1(sum(cs(s, c) for c in keep)) for s in SEEDS]))
    full = delta(test)
    loo = {c: delta([x for x in test if x != c]) for c in test}
    n = len(test)
    jse = math.sqrt((n - 1) / n * sum((v - np.mean(list(loo.values()))) ** 2 for v in loo.values()))
    per_city = {c: float(np.mean([f1(ce(s, c)) - f1(cs(s, c)) for s in SEEDS])) for c in test}
    return {"delta": full, "left_out": loo, "left_out_range": [min(loo.values()), max(loo.values())], "jackknife_se": jse,
            "per_city_delta": per_city, "cities_same_sign": int(sum(np.sign(v) == np.sign(full) for v in per_city.values()))}


def analyse(runs: dict, cities_dir: Path, step9_runs: Path) -> dict:
    val = lambda a, c, key="f1", thr="at_selected": [runs[a, s]["test"][c][thr][key] or 0.0 for s in SEEDS]
    E, S = "p10_fusion_dropout", "p10_unet_sar"
    out = {"arms": {a: {thr: {c: {key: describe(val(a, c, key, thr)) for key in METRICS} for c in CONDS[a]}
                        for thr in ("at_selected", "at_0.5")} for a in CONDS},
           "selected_step": {a: [runs[a, s]["selected_step"] for s in SEEDS] for a in CONDS},
           "selected_threshold": {a: {c: [runs[a, s]["test"][c]["selected_threshold"] for s in SEEDS] for c in CONDS[a]} for a in CONDS}}
    out["primary"] = family({"E@no_opt - SAR-only": (val(S, "full"), val(E, "no_opt"))}, "E without optical")
    out["secondary"] = family({"E: image present - absent (Step 9 H1)": (val(E, "no_opt"), val(E, "no_o2")),
                               "E@no_opt - SAR-only, threshold 0.5": (val(S, "full", thr="at_0.5"), val(E, "no_opt", thr="at_0.5"))},
                              "higher/lower")
    old = lambda a, c: [json.loads((step9_runs / f"{a}_seed{s}" / "metrics.json").read_text(encoding="utf-8"))["test"][c]["at_selected"]["f1"]
                        or 0.0 for s in range(1, 11)]
    out["descriptive_20_seeds"] = paired(old("p6_unet_sar", "full") + val(S, "full"), old("p9_fusion_dropout", "no_opt") + val(E, "no_opt"))
    out["city_level"] = city_level(cities_dir, runs)
    return out


def report(out: dict) -> str:
    L = ["# Step 10: confirmatory replication on seeds 11-20", "",
         "Pre-registered in `STEP10_PREREGISTRATION.md`. Generated by `analyze_step10.py`; numbers in `summary.json`.", ""]
    L += family_table(out["primary"], "## Primary: E without optical input minus the SAR-only U-Net (single test)")
    L += family_table(out["secondary"], "## Secondary (Holm over 2)")
    r = out["descriptive_20_seeds"]
    L += [f"Descriptive, all 20 seeds (seeds 1-10 selected the hypothesis): {r['mean_diff']:+.3f} "
          f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}], {r['b_gt_a']}/20 positive.", ""]
    c = out["city_level"]
    L += ["## City level (exploratory; primary contrast)", "",
          f"Δ {c['delta']:+.3f}; one city left out {c['left_out_range'][0]:+.3f} to {c['left_out_range'][1]:+.3f}; "
          f"jackknife SE {c['jackknife_se']:.3f}; cities with the same sign {c['cities_same_sign']}/10 (unweighted).", "",
          "| city | Δ (mean over seeds) | pooled Δ without it |", "|---|---|---|"]
    L += [f"| {k} | {v:+.3f} | {c['left_out'][k]:+.3f} |" for k, v in c["per_city_delta"].items()] + [""]
    for thr, name in (("at_selected", "validation-selected threshold"), ("at_0.5", "threshold 0.5")):
        L += [f"## Test metrics at the {name}, mean ± SD over seeds 11-20", "",
              "| arm | condition | F1 | precision | recall | IoU | false-alarm rate |", "|---|---|---|---|---|---|---|"]
        for a, conds in CONDS.items():
            for cond in conds:
                e = out["arms"][a][thr][cond]
                L.append(f"| {NAMES[a]} | {cond} | " + " | ".join(f"{e[k]['mean']:.3f} ± {e[k]['sd']:.3f}" for k in METRICS) + " |")
        L.append("")
    return "\n".join(L) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=ROOT / "experiments" / "robust")
    p.add_argument("--step9-runs", type=Path, default=ROOT / "experiments" / "robust")
    p.add_argument("--cities", type=Path, default=Path(__file__).resolve().parent / "cities")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    a = p.parse_args()
    if (a.out / "summary.json").exists():
        raise SystemExit(f"{a.out / 'summary.json'} exists; refusing to overwrite")
    out = analyse(load(a.runs), a.cities, a.step9_runs)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (a.out / "STEP10_REPORT.md").write_text(report(out), encoding="utf-8")
    print(f"wrote {a.out / 'summary.json'} and STEP10_REPORT.md")


if __name__ == "__main__":
    main()
