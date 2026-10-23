"""Step 6 analysis (RRD RQ1, RQ4): baselines and the asynchronous configuration, fixed before any Step 6 test output.

Reads experiments/robust/{arm}_seed{1..10}/metrics.json (written by train_robust.py for A and B, train_rrd.py for the
Step 6 arms). Test F1 at each run's validation-selected threshold for the condition (Phase 2's rule). Writes
experiments/robust/phase6/summary.json and tables.md; refuses to overwrite them.

Families (paired over seeds 1-10; paired t-test, Holm within each family; Wilcoxon and Holm-Wilcoxon alongside):
    primary_rq4  full input: A - SAR-only U-Net, A - optical-only U-Net, A - input-fusion U-Net          (Holm over 3)
    primary_rq1  SAR + historical optical vs SAR only: B@no_o2 - SAR-only, async@no_o2 - SAR-only       (Holm over 2)
    secondary_fusion     full input: fusion - optical-only, fusion - SAR-only, optical-only - SAR-only
    secondary_async      async@no_o2 - B@no_o2; B@no_o2 - B@no_opt (historical optical removed, within B)

Usage:
    .venv/Scripts/python.exe analyze_phase6.py
"""

import json

from analyze_phase3 import describe, fmt_p, holm, ms, paired
from src.training.train_robust import CONDITIONS, PROJECT_ROOT

R = PROJECT_ROOT / "experiments" / "robust"
OUT = R / "phase6"
SEEDS = range(1, 11)
ARMS = {"p2_full": "A DS-UNet, full-only", "p2_dropout": "B DS-UNet, modality dropout", "p6_unet_sar": "SAR-only U-Net",
        "p6_unet_opt": "optical-only U-Net", "p6_unet_fusion": "input-fusion U-Net", "p6_async": "async DS-UNet (no_o2 only)"}
METRICS = ("f1", "precision", "recall", "iou", "false_alarm_rate")


def load():
    runs = {}
    for arm in ARMS:
        for s in SEEDS:
            m = json.loads((R / f"{arm}_seed{s}" / "metrics.json").read_text(encoding="utf-8"))
            runs[arm, s] = m
    cities = {tuple(m["test_cities"]) for m in runs.values()} | {("val",) + tuple(m["val_cities"]) for m in runs.values()}
    assert len(cities) == 2, "all runs must share the test and validation cities"
    return runs


def main() -> None:
    if (OUT / "summary.json").exists():
        raise SystemExit("phase6/summary.json exists; refusing to overwrite")
    runs = load()
    # an undefined rate (zero denominator, e.g. no predicted change) counts as 0, as F1 does
    val = lambda arm, c, k="f1", thr="at_selected": [runs[arm, s]["test"][c][thr][k] or 0.0 for s in SEEDS]

    def family(pairs, label=None):
        rows = {k: paired(a, b) for k, (a, b) in pairs.items()}
        for k, h in zip(rows, holm([r["t_p"] for r in rows.values()])):
            rows[k]["holm_p"] = h
        for k, h in zip(rows, holm([r["wilcoxon_p"] for r in rows.values()])):
            rows[k]["holm_wilcoxon_p"] = h
        if label:
            for r in rows.values():
                r["label"] = (f"{label[0]} better" if r["holm_p"] < 0.05 and r["mean_diff"] > 0 else
                              f"{label[0]} worse" if r["holm_p"] < 0.05 and r["mean_diff"] < 0 else "no detectable difference")
        return rows

    # each pair is (a, b) and the contrast is b - a
    out = {"arms": {arm: {thr: {c: {k: describe(val(arm, c, k, thr)) for k in METRICS} for c in CONDITIONS}
                          for thr in ("at_selected", "at_0.5")} for arm in ARMS},
           "selected_step": {arm: [runs[arm, s]["selected_step"] for s in SEEDS] for arm in ARMS},
           "selected_threshold": {arm: {c: [runs[arm, s]["test"][c]["selected_threshold"] for s in SEEDS] for c in CONDITIONS} for arm in ARMS}}
    out["primary_rq4"] = family({"A - SAR-only": (val("p6_unet_sar", "full"), val("p2_full", "full")),
                                 "A - optical-only": (val("p6_unet_opt", "full"), val("p2_full", "full")),
                                 "A - input fusion": (val("p6_unet_fusion", "full"), val("p2_full", "full"))}, ("DS-UNet",))
    out["primary_rq1"] = family({"B@no_o2 - SAR-only": (val("p6_unet_sar", "full"), val("p2_dropout", "no_o2")),
                                 "async@no_o2 - SAR-only": (val("p6_unet_sar", "full"), val("p6_async", "no_o2"))},
                                ("SAR + historical optical",))
    out["secondary_fusion"] = family({"fusion - optical-only": (val("p6_unet_opt", "full"), val("p6_unet_fusion", "full")),
                                      "fusion - SAR-only": (val("p6_unet_sar", "full"), val("p6_unet_fusion", "full")),
                                      "optical-only - SAR-only": (val("p6_unet_sar", "full"), val("p6_unet_opt", "full"))})
    out["secondary_async"] = family({"async@no_o2 - B@no_o2": (val("p2_dropout", "no_o2"), val("p6_async", "no_o2")),
                                     "B@no_o2 - B@no_opt": (val("p2_dropout", "no_opt"), val("p2_dropout", "no_o2"))})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (OUT / "tables.md").write_text(tables(out), encoding="utf-8")
    print("phase6 summary written")


def family_table(rows, title):
    L = [title, "", "| contrast | mean Δ | 95% CI | d_z | Δ > 0 | Holm p (t) | Holm p (Wilcoxon) | label |", "|---|---|---|---|---|---|---|---|"]
    for k, r in rows.items():
        L.append(f"| {k} | {r['mean_diff']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['dz']:+.2f} | {r['b_gt_a']}/10 | "
                 f"{fmt_p(r['holm_p'])} | {fmt_p(r['holm_wilcoxon_p'])} | {r.get('label', '')} |")
    return L + [""]


def tables(out) -> str:
    L = family_table(out["primary_rq4"], "### PRIMARY (RQ4): full input, test F1 at the validation-selected threshold, Holm over 3")
    L += family_table(out["primary_rq1"], "### PRIMARY (RQ1): SAR + historical optical vs SAR only, Holm over 2")
    L += family_table(out["secondary_fusion"], "### Secondary: fusion and unimodal baselines, full input (own Holm family)")
    L += family_table(out["secondary_async"], "### Secondary: asynchronous configurations (own Holm family)")
    for thr, name in (("at_selected", "validation-selected threshold"), ("at_0.5", "threshold 0.5")):
        L += [f"### Test metrics at the {name}, mean ± SD over 10 seeds", "",
              "| arm | condition | F1 | precision | recall | IoU | false-alarm rate |", "|---|---|---|---|---|---|---|"]
        for arm, name_ in ARMS.items():
            for c in CONDITIONS:
                e = out["arms"][arm][thr][c]
                L.append(f"| {name_} | {c} | " + " | ".join(ms(e[k]) for k in METRICS) + " |")
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
