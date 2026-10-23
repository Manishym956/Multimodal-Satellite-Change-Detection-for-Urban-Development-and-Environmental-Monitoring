"""Step 7 analysis (RRD RQ3, polarization), fixed before any Step 7 test output.

Reads experiments/robust/phase7/runs/{arm}_seed{s}.npz (eval_phase7.py). Counts are pooled over CITIES (the 9 test cities
with VH at both dates; Las Vegas has none at date 1) at each checkpoint's frozen validation threshold for full input.
Writes experiments/robust/phase7/summary.json and tables.md; refuses to overwrite.

Families (paired over seeds 1-10; paired t-test, Holm within family; Wilcoxon alongside):
    primary_rq3        SAR-only U-Net: VV+VH - VV, VH - VV (F1)                          (Holm over 2)
    secondary_dsunet   DS-UNet (full-only recipe): VV+VH - VV, VH - VV (F1)               (Holm over 2)
    secondary_far      SAR-only U-Net, false-alarm rate VV+VH - VV per WorldCover class with >= 1% of the pooled
                       pixels                                                             (Holm over those classes)
Descriptive: F1, precision, recall, IoU, false-alarm rate per arm on the 9 cities and on all 10 (Las Vegas flagged:
its VH t1 band is empty); per-class false-alarm rate and recall per arm.

Usage:
    .venv/Scripts/python.exe analyze_phase7.py
"""

import json

import numpy as np

from analyze_phase3 import describe, fmt_p, holm, ms, paired
from eval_phase7 import ARMS, OUT, SEEDS, WC_CLASSES

CITIES = ("brasilia", "montpellier", "norcia", "rio", "saclay_w", "valencia", "dubai", "milano", "chongqing")
ALL10 = CITIES + ("lasvegas",)
WC_NAMES = {0: "no data", 10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland", 50: "built-up", 60: "bare",
            70: "snow and ice", 80: "water", 90: "herbaceous wetland", 95: "mangroves", 100: "moss and lichen"}
METRICS = ("f1", "precision", "recall", "iou", "false_alarm_rate")


def rates(tp, fp, fn, tn):
    p, r = (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)
    return {"f1": 2 * p * r / (p + r) if p + r else 0.0, "precision": p, "recall": r,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0, "false_alarm_rate": fp / (fp + tn) if fp + tn else 0.0}


def main() -> None:
    if (OUT / "summary.json").exists():
        raise SystemExit("phase7/summary.json exists; refusing to overwrite")
    runs = {}
    for arm in ARMS:
        for s in SEEDS:
            meta = json.loads((OUT / "runs" / f"{arm}_seed{s}.json").read_text(encoding="utf-8"))
            assert all(meta["anchors"].values())
            runs[arm, s] = (meta, np.load(OUT / "runs" / f"{arm}_seed{s}.npz"))

    def pooled(arm, s, cities, cond="full"):
        meta, z = runs[arm, s]
        k = meta["thresholds"][cond]
        return [float(x) for x in sum(z[f"{c}__{cond}"][k] for c in cities)]

    def metric(arm, k="f1", cities=CITIES):
        return [rates(*pooled(arm, s, cities))[k] for s in SEEDS]

    def wc(arm, cls, k):
        return [rates(*[float(x) for x in sum(runs[arm, s][1][f"{c}__wc{cls}"] for c in CITIES)])[k] for s in SEEDS]

    def family(pairs):
        rows = {k: paired(a, b) for k, (a, b) in pairs.items()}
        for k, h in zip(rows, holm([r["t_p"] for r in rows.values()])):
            rows[k]["holm_p"] = h
        for k, h in zip(rows, holm([r["wilcoxon_p"] for r in rows.values()])):
            rows[k]["holm_wilcoxon_p"] = h
        for k, r in rows.items():
            first = k.split(" - ")[0]
            r["label"] = (f"{first} higher" if r["holm_p"] < 0.05 and r["mean_diff"] > 0 else
                          f"{first} lower" if r["holm_p"] < 0.05 and r["mean_diff"] < 0 else "no detectable difference")
        return rows

    # pixel share of each class over the 9 cities (labels only; identical for every run)
    _, z0 = runs["p6_unet_sar", 1]
    share = {cls: float(sum(z0[f"{c}__wc{cls}"].sum() for c in CITIES)) for cls in WC_CLASSES}
    total = sum(share.values())
    share = {cls: v / total for cls, v in share.items()}
    tested = [cls for cls in WC_CLASSES if cls != 0 and share[cls] >= 0.01]

    out = {"cities": list(CITIES), "arms": {arm: {scope: {k: describe(metric(arm, k, cities)) for k in METRICS}
                                                  for scope, cities in (("9_cities", CITIES), ("10_cities", ALL10))} for arm in ARMS},
           "worldcover_share": {WC_NAMES[c]: share[c] for c in WC_CLASSES},
           "worldcover": {arm: {WC_NAMES[c]: {k: describe(wc(arm, c, k)) for k in ("false_alarm_rate", "recall")} for c in tested}
                          for arm in ARMS}}
    out["primary_rq3"] = family({"VV+VH - VV": (metric("p6_unet_sar"), metric("p7_unet_vvvh")),
                                 "VH - VV": (metric("p6_unet_sar"), metric("p7_unet_vh"))})
    out["secondary_dsunet"] = family({"VV+VH - VV": (metric("p2_full"), metric("p7_dsunet_vvvh")),
                                      "VH - VV": (metric("p2_full"), metric("p7_dsunet_vh"))})
    out["secondary_far"] = family({f"VV+VH - VV, {WC_NAMES[c]}": (wc("p6_unet_sar", c, "false_alarm_rate"),
                                                                 wc("p7_unet_vvvh", c, "false_alarm_rate")) for c in tested})
    (OUT / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (OUT / "tables.md").write_text(tables(out), encoding="utf-8")
    print("phase7 summary written")


NAMES = {"p6_unet_sar": "SAR-only U-Net, VV", "p7_unet_vh": "SAR-only U-Net, VH", "p7_unet_vvvh": "SAR-only U-Net, VV+VH",
         "p2_full": "DS-UNet, VV", "p7_dsunet_vh": "DS-UNet, VH", "p7_dsunet_vvvh": "DS-UNet, VV+VH"}


def family_table(rows, title):
    L = [title, "", "| contrast | mean Δ | 95% CI | d_z | Δ > 0 | Holm p (t) | Holm p (Wilcoxon) | label |", "|---|---|---|---|---|---|---|---|"]
    for k, r in rows.items():
        L.append(f"| {k} | {r['mean_diff']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['dz']:+.2f} | {r['b_gt_a']}/10 | "
                 f"{fmt_p(r['holm_p'])} | {fmt_p(r['holm_wilcoxon_p'])} | {r['label']} |")
    return L + [""]


def tables(out) -> str:
    L = family_table(out["primary_rq3"], "### PRIMARY (RQ3): SAR-only U-Net, test F1 on 9 cities, frozen threshold, Holm over 2")
    L += family_table(out["secondary_dsunet"], "### Secondary: DS-UNet, test F1 on 9 cities (own Holm family)")
    L += family_table(out["secondary_far"], "### Secondary: SAR-only U-Net, false-alarm rate by WorldCover class (own Holm family)")
    for scope in ("9_cities", "10_cities"):
        L += [f"### Test metrics, full input, {scope.replace('_', ' ')}, mean ± SD over 10 seeds", "",
              "| arm | F1 | precision | recall | IoU | false-alarm rate |", "|---|---|---|---|---|---|"]
        L += [f"| {NAMES[a]} | " + " | ".join(ms(out["arms"][a][scope][k]) for k in METRICS) + " |" for a in NAMES] + [""]
    L += ["### False-alarm rate and recall by WorldCover class (9 cities, full input)", "",
          "| arm | " + " | ".join(out["worldcover"]["p6_unet_sar"]) + " |", "|---|" + "---|" * len(out["worldcover"]["p6_unet_sar"])]
    for a in NAMES:
        L.append(f"| {NAMES[a]} | " + " | ".join(f"FAR {e['false_alarm_rate']['mean']:.3f}, R {e['recall']['mean']:.3f}"
                                                for e in out["worldcover"][a].values()) + " |")
    L += ["", "Pixel share by class: " + ", ".join(f"{k} {v:.3f}" for k, v in out["worldcover_share"].items() if v > 0), ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
