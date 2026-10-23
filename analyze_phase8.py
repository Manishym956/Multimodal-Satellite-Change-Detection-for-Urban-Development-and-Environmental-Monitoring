"""Step 8 analysis (RRD RQ1, RQ2, RQ5), fixed before any Step 8 test output.

Reads experiments/robust/phase8/runs/{arm}_seed{s}.npz (eval_phase8.py) and datasets/oscd_s2_gee/manifest.json.
Primary city set: test cities whose selected Earth Engine scene has cloud fraction <= 0.10 over the city box at every gap
(decided from the manifest only). Counts are pooled over those cities at each run's frozen threshold (no_o2 for the
DS-UNet arms, full for the SAR-only U-Net). Writes experiments/robust/phase8/summary.json and tables.md; refuses to overwrite.

    primary_rq2        B: per-seed least-squares slope of F1 against the gap in months (1, 3, 6, 12, 18); one-sample
                       t-test of the 10 slopes against 0 (two-sided), Wilcoxon alongside
    secondary_B_vs_sar      B@gapNN - SAR-only, F1                                        (Holm over 5)
    secondary_async_vs_sar  async@gapNN - SAR-only, F1                                    (Holm over 5)
    secondary_async_slope   async slope, as the primary
    secondary_source        B@gee_t1 - B@oscd (same date, Earth Engine vs OSCD processing); B@oscd - SAR-only
    rq5_landcover      false-alarm rate, B@oscd - SAR-only, per WorldCover class with >= 1% of the pooled pixels
                                                                                           (Holm over those classes)
Descriptive: every metric per arm and tag; the same on all cities with a scene (sensitivity); F1 per tag pooled
separately over cities whose gap image precedes OSCD date 1 and those whose image does not; per-class false-alarm rate
and recall for every arm, tag and the full-input pass.

Usage:
    .venv/Scripts/python.exe analyze_phase8.py
"""

import json

import numpy as np
from scipy import stats

from analyze_phase3 import describe, fmt_p, holm, ms, paired
from analyze_phase7 import WC_NAMES, rates
from eval_phase7 import WC_CLASSES
from eval_phase8 import ARMS, OUT, S2_ROOT, SEEDS, TAGS

GAPS = {"gap01": 1, "gap03": 3, "gap06": 6, "gap12": 12, "gap18": 18}
METRICS = ("f1", "precision", "recall", "iou", "false_alarm_rate")
MAX_CLOUD = 0.10


def city_sets(test_cities):
    man = json.loads((S2_ROOT / "manifest.json").read_text(encoding="utf-8"))
    row = {(r["city"], r["tag"]): r for r in man}
    clear = lambda c, t: (c, t) in row and row[c, t].get("day") and row[c, t]["cloud"] <= MAX_CLOUD
    primary = [c for c in test_cities if all(clear(c, t) for t in GAPS)]
    available = [c for c in test_cities if all((c, t) in row and row[c, t].get("day") for t in GAPS)]
    before = {t: [c for c in test_cities if (c, t) in row and row[c, t].get("day") and row[c, t]["day"] <= row[c, t]["oscd_t1"]]
              for t in GAPS}
    return primary, available, before, {f"{c}__{t}": row[c, t] for c, t in row if c in test_cities}


def slope(f1_by_gap):
    x = np.array(list(GAPS.values()), dtype=float)
    return float(np.polyfit(x, np.array(f1_by_gap), 1)[0])


def main() -> None:
    if (OUT / "summary.json").exists():
        raise SystemExit("phase8/summary.json exists; refusing to overwrite")
    runs = {}
    for arm in ARMS:
        for s in SEEDS:
            meta = json.loads((OUT / "runs" / f"{arm}_seed{s}.json").read_text(encoding="utf-8"))
            assert all(meta["anchors"].values())
            runs[arm, s] = (meta, np.load(OUT / "runs" / f"{arm}_seed{s}.npz"))
    test_cities = runs["p2_dropout", 1][0]["test_cities"]
    primary, available, before, scenes = city_sets(test_cities)

    def counts(arm, s, tag, cities, cls=None):
        meta, z = runs[arm, s]
        sar = arm == "p6_unet_sar"
        tag = "oscd" if sar else tag
        k = meta["thresholds"]["full" if sar or tag == "full" else "no_o2"]
        key = (lambda c: f"{c}__{tag}__wc{cls}") if cls is not None else (lambda c: f"{c}__{tag}")
        return [float(x) for x in sum((z[key(c)] if cls is not None else z[key(c)][k]) for c in cities)]

    met = lambda arm, tag, k="f1", cities=primary, cls=None: [rates(*counts(arm, s, tag, cities, cls))[k] for s in SEEDS]

    def family(pairs, prefix=None):
        rows = {k: paired(a, b) for k, (a, b) in pairs.items()}
        for k, h in zip(rows, holm([r["t_p"] for r in rows.values()])):
            rows[k]["holm_p"] = h
        for k, h in zip(rows, holm([r["wilcoxon_p"] for r in rows.values()])):
            rows[k]["holm_wilcoxon_p"] = h
        for k, r in rows.items():
            first = prefix or k.split(" - ")[0]
            r["label"] = (f"{first} higher" if r["holm_p"] < 0.05 and r["mean_diff"] > 0 else
                          f"{first} lower" if r["holm_p"] < 0.05 and r["mean_diff"] < 0 else "no detectable difference")
        return rows

    def slope_test(arm, cities=primary):
        sl = [slope([rates(*counts(arm, s, t, cities))["f1"] for t in GAPS]) for s in SEEDS]
        d = np.array(sl)
        half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        return {"slopes_f1_per_month": sl, "mean": float(d.mean()), "ci95": [float(d.mean() - half), float(d.mean() + half)],
                "t_p": float(stats.ttest_1samp(d, 0).pvalue), "wilcoxon_p": float(stats.wilcoxon(d).pvalue),
                "n_negative": int((d < 0).sum()),
                "label": ("F1 falls with the gap" if stats.ttest_1samp(d, 0).pvalue < 0.05 and d.mean() < 0 else
                          "F1 rises with the gap" if stats.ttest_1samp(d, 0).pvalue < 0.05 and d.mean() > 0 else "no detectable effect of the gap")}

    # pixel share of each WorldCover class over the primary cities (labels only)
    _, z0 = runs["p6_unet_sar", 1]
    share = {cls: float(sum(z0[f"{c}__oscd__wc{cls}"].sum() for c in primary)) for cls in WC_CLASSES}
    total = sum(share.values())
    share = {cls: v / total for cls, v in share.items()}
    tested = [cls for cls in WC_CLASSES if cls != 0 and share[cls] >= 0.01]
    dsunet_arms = [a for a in ARMS if a != "p6_unet_sar"]

    out = {"primary_cities": primary, "available_cities": available, "max_cloud": MAX_CLOUD, "scenes": scenes,
           "image_before_oscd_t1": before,
           "arms": {a: {t: {k: describe(met(a, t, k)) for k in METRICS} for t in (TAGS if a != "p6_unet_sar" else ("oscd",))}
                    for a in ARMS},
           "arms_full_input": {a: {k: describe(met(a, "full", k)) for k in METRICS} for a in dsunet_arms},
           "sensitivity_all_cities": {a: {t: describe(met(a, t, "f1", available)) for t in TAGS} for a in dsunet_arms},
           "by_precedence": {a: {t: {grp: describe(met(a, t, "f1", cs)) if cs else None
                                     for grp, cs in (("before_t1", [c for c in primary if c in before[t]]),
                                                     ("after_t1", [c for c in primary if c not in before[t]]))}
                                 for t in GAPS} for a in dsunet_arms},
           "worldcover_share": {WC_NAMES[c]: share[c] for c in WC_CLASSES},
           "worldcover": {a: {t: {WC_NAMES[c]: {k: describe(met(a, t, k, primary, c)) for k in ("false_alarm_rate", "recall")}
                                  for c in tested} for t in (("oscd", "full") + tuple(GAPS) if a != "p6_unet_sar" else ("oscd",))}
                          for a in ARMS}}
    sar = met("p6_unet_sar", "oscd")
    out["primary_rq2"] = slope_test("p2_dropout")
    out["secondary_B_vs_sar"] = family({f"B@{t} - SAR-only": (sar, met("p2_dropout", t)) for t in GAPS}, "B")
    out["secondary_async_vs_sar"] = family({f"async@{t} - SAR-only": (sar, met("p6_async", t)) for t in GAPS}, "async")
    out["secondary_async_slope"] = slope_test("p6_async")
    out["secondary_source"] = family({"B@gee_t1 - B@oscd": (met("p2_dropout", "oscd"), met("p2_dropout", "gee_t1")),
                                      "B@oscd - SAR-only": (sar, met("p2_dropout", "oscd"))})
    out["sensitivity_slope_all_cities"] = slope_test("p2_dropout", available)
    out["rq5_landcover"] = family({f"B@oscd - SAR-only, {WC_NAMES[c]}": (met("p6_unet_sar", "oscd", "false_alarm_rate", primary, c),
                                                                        met("p2_dropout", "oscd", "false_alarm_rate", primary, c))
                                   for c in tested}, "B")
    (OUT / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (OUT / "tables.md").write_text(tables(out), encoding="utf-8")
    print("phase8 summary written")


NAMES = {"p2_dropout": "B dropout", "p6_async": "async", "p2_full": "A full-only", "p6_unet_sar": "SAR-only U-Net"}


def family_table(rows, title):
    L = [title, "", "| contrast | mean Δ | 95% CI | d_z | Δ > 0 | Holm p (t) | Holm p (Wilcoxon) | label |", "|---|---|---|---|---|---|---|---|"]
    for k, r in rows.items():
        L.append(f"| {k} | {r['mean_diff']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['dz']:+.2f} | {r['b_gt_a']}/10 | "
                 f"{fmt_p(r['holm_p'])} | {fmt_p(r['holm_wilcoxon_p'])} | {r['label']} |")
    return L + [""]


def slope_line(r, title):
    return [title, "", f"mean slope {r['mean'] * 12:+.4f} F1 per year ({r['mean']:+.5f} per month), 95% CI "
            f"[{r['ci95'][0] * 12:+.4f}, {r['ci95'][1] * 12:+.4f}] per year; negative in {r['n_negative']}/10 seeds; "
            f"t p {fmt_p(r['t_p'])}, Wilcoxon p {fmt_p(r['wilcoxon_p'])}: **{r['label']}**", ""]


def tables(out) -> str:
    L = [f"Primary cities (scene cloud <= {out['max_cloud']} at every gap): {', '.join(out['primary_cities'])}", ""]
    L += slope_line(out["primary_rq2"], "### PRIMARY (RQ2): B, slope of test F1 against the gap (no_o2, frozen threshold)")
    L += family_table(out["secondary_B_vs_sar"], "### Secondary (RQ1 by gap): B − SAR-only, Holm over 5")
    L += family_table(out["secondary_async_vs_sar"], "### Secondary: async − SAR-only, Holm over 5")
    L += slope_line(out["secondary_async_slope"], "### Secondary: async, slope against the gap")
    L += family_table(out["secondary_source"], "### Secondary: source control and native historical optical (own Holm family)")
    L += slope_line(out["sensitivity_slope_all_cities"], "### Sensitivity: B slope on all cities with a scene at every gap")
    L += family_table(out["rq5_landcover"], "### RQ5: false-alarm rate by WorldCover class, B@oscd − SAR-only (own Holm family)")
    L += ["### Test F1 by historical image (no_o2), mean ± SD over 10 seeds", "",
          "| arm | " + " | ".join(TAGS) + " |", "|---|" + "---|" * len(TAGS)]
    for a in ("p2_dropout", "p6_async", "p2_full"):
        L.append(f"| {NAMES[a]} | " + " | ".join(ms(out["arms"][a][t]["f1"]) for t in TAGS) + " |")
    L += [f"| SAR-only U-Net (reference) | {ms(out['arms']['p6_unet_sar']['oscd']['f1'])} |", ""]
    L += ["Scenes: " + "; ".join(f"{k} {v.get('day')} (cloud {v.get('cloud', float('nan')):.3f})" for k, v in out["scenes"].items()
                                 if v.get("day")), ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
