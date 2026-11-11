"""Step 11 analysis (pre-registered in STEP11_PREREGISTRATION.md): real cloudy post-change scenes.

Reads runs/{arm}_seed{1..10}.npz (eval_real_cloud.py). Each model at its frozen clean-validation threshold (A, B: the
checkpoint's `full` threshold, as Phases 3-5; SAR-only S: its frozen threshold). Per seed, counts are pooled over the
cities of a set and F1 is computed from the pooled counts. Paired over seeds 1-10: paired t-test, two-sided, Holm within
each family, with mean Δ, 95% CI, d_z, seeds with Δ > 0 and Wilcoxon alongside.
    primary (7 cloudy cities, Holm over 3)
        B real_rb95 - B real          masking real cloud helps the dropout model
        B real_rb95 - S               the dropout model with a mask against the SAR-only fallback
        B real - A real               dropout degrades less than full-input training on real cloud
    secondary (7 cloudy cities, Holm over 4)
        B real_bal - B real; B real_rb95 - B missing; A real - S; B control - B oscd
    overcast (3 cities, Holm over 3) and all 10 cities (Holm over 3): the three primary contrasts
    descriptive: F1, precision, recall, false-alarm rate of every arm and condition per set; per-city F1; per-city Δ of
        the primary contrasts and one-city-left-out ranges (exploratory)
Writes summary.json and STEP11_REPORT.md next to this file; refuses to overwrite.

Usage (repository root):  .venv/Scripts/python.exe experiments/robust/realcloud/analyze_real_cloud.py [--runs D --out D]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "robust" / "step9"))
from analyze_phase3 import describe  # noqa: E402
from analyze_step9 import family, family_table  # noqa: E402
from src.evaluation.metrics import rates_from_counts  # noqa: E402

SEEDS = range(1, 11)
CLOUDY = ("brasilia", "norcia", "saclay_w", "valencia", "dubai", "milano", "chongqing")
OVERCAST = ("montpellier", "rio", "lasvegas")
SETS = {"cloudy": CLOUDY, "overcast": OVERCAST, "all": CLOUDY + OVERCAST}
CONDS = ("oscd", "control", "real", "real_bal", "real_rb95", "real_rb99", "control_rb95", "missing")
METRICS = ("f1", "precision", "recall", "false_alarm_rate")


def rates(c) -> dict:
    r = rates_from_counts({"tp": int(c[0]), "fp": int(c[1]), "fn": int(c[2]), "tn": int(c[3])})
    return {k: r[k] or 0.0 for k in METRICS}


def load(runs: Path):
    counts, tau = {}, {}
    for arm in ("p2_full", "p2_dropout", "p6_unet_sar"):
        for s in SEEDS:
            meta = json.loads((runs / f"{arm}_seed{s}.json").read_text(encoding="utf-8"))
            assert all(meta["anchors"].values()), f"{arm} {s}: anchor not passed"
            counts[arm, s], tau[arm, s] = np.load(runs / f"{arm}_seed{s}.npz"), meta["tau_clean"]
    return counts, tau


def analyse(counts, tau) -> dict:
    def at(arm, s, city, cond):  # S sees only SAR: its counts do not depend on the optical condition
        return counts[arm, s][f"{city}__{'oscd' if arm == 'p6_unet_sar' else cond}"][tau[arm, s]]

    val = lambda arm, cond, cities, k="f1": [rates(sum(at(arm, s, c, cond) for c in cities))[k] for s in SEEDS]
    A, B, S = "p2_full", "p2_dropout", "p6_unet_sar"

    def primary(cities):
        return {"B real_rb95 - B real": (val(B, "real", cities), val(B, "real_rb95", cities)),
                "B real_rb95 - SAR-only": (val(S, "oscd", cities), val(B, "real_rb95", cities)),
                "B real - A real": (val(A, "real", cities), val(B, "real", cities))}

    out = {"primary": family(primary(CLOUDY), "higher/lower"),
           "secondary": family({"B real_bal - B real": (val(B, "real", CLOUDY), val(B, "real_bal", CLOUDY)),
                                "B real_rb95 - B missing": (val(B, "missing", CLOUDY), val(B, "real_rb95", CLOUDY)),
                                "A real - SAR-only": (val(S, "oscd", CLOUDY), val(A, "real", CLOUDY)),
                                "B control - B oscd": (val(B, "oscd", CLOUDY), val(B, "control", CLOUDY))}, "higher/lower"),
           "overcast": family(primary(OVERCAST), "higher/lower"),
           "all": family(primary(SETS["all"]), "higher/lower"),
           "arms": {set_: {arm: {cond: {k: describe(val(arm, cond, cities, k)) for k in METRICS} for cond in CONDS}
                           for arm in (A, B, S)} for set_, cities in SETS.items()},
           "per_city_f1": {arm: {c: {cond: float(np.mean([rates(at(arm, s, c, cond))["f1"] for s in SEEDS])) for cond in CONDS}
                                 for c in SETS["all"]} for arm in (A, B, S)}}
    city = {}
    for name in primary(CLOUDY):
        loo = {}
        for left in CLOUDY:
            keep = [c for c in CLOUDY if c != left]
            pa, pb = primary(keep)[name]
            loo[left] = float(np.mean(np.array(pb) - np.array(pa)))
        city[name] = {"left_out": loo, "left_out_range": [min(loo.values()), max(loo.values())]}
    out["city_level"] = city
    return out


def report(out: dict) -> str:
    L = ["# Step 11: real cloudy post-change scenes", "",
         "Pre-registered in `STEP11_PREREGISTRATION.md`. Generated by `analyze_real_cloud.py`; numbers in `summary.json`.", ""]
    L += family_table(out["primary"], "## Primary: 7 cloudy cities, F1 at the frozen clean thresholds (Holm over 3)")
    L += family_table(out["secondary"], "## Secondary: 7 cloudy cities (Holm over 4)")
    L += family_table(out["overcast"], "## Overcast cities: montpellier, rio, lasvegas (Holm over 3)")
    L += family_table(out["all"], "## All 10 cities (Holm over 3)")
    L += ["## City level (exploratory): primary contrasts with one cloudy city left out", "", "| contrast | range |", "|---|---|"]
    L += [f"| {k} | {v['left_out_range'][0]:+.3f} to {v['left_out_range'][1]:+.3f} |" for k, v in out["city_level"].items()] + [""]
    names = {"p2_full": "A", "p2_dropout": "B", "p6_unet_sar": "SAR-only"}
    for set_ in SETS:
        L += [f"## Test metrics, {set_} cities, mean ± SD over 10 seeds", "",
              "| arm | condition | F1 | precision | recall | false-alarm rate |", "|---|---|---|---|---|---|"]
        for arm, n in names.items():
            for cond in (("oscd",) if arm == "p6_unet_sar" else CONDS):
                e = out["arms"][set_][arm][cond]
                L.append(f"| {n} | {cond} | " + " | ".join(f"{e[k]['mean']:.3f} ± {e[k]['sd']:.3f}" for k in METRICS) + " |")
        L.append("")
    L += ["## Per-city F1 (mean over seeds)", "", "| city | arm | " + " | ".join(CONDS) + " |", "|---|---|" + "---|" * len(CONDS)]
    for c in SETS["all"]:
        for arm, n in names.items():
            L.append(f"| {c} | {n} | " + " | ".join(f"{out['per_city_f1'][arm][c][k]:.3f}" for k in CONDS) + " |")
    return "\n".join(L) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=HERE / "runs")
    p.add_argument("--out", type=Path, default=HERE)
    a = p.parse_args()
    if (a.out / "summary.json").exists():
        raise SystemExit(f"{a.out / 'summary.json'} exists; refusing to overwrite")
    out = analyse(*load(a.runs))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (a.out / "STEP11_REPORT.md").write_text(report(out), encoding="utf-8")
    print(f"wrote {a.out / 'summary.json'} and STEP11_REPORT.md")


if __name__ == "__main__":
    main()
