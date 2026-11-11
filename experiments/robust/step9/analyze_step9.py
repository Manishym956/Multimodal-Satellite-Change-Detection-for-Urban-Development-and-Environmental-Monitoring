"""Step 9 analysis (pre-registered in STEP9_PREREGISTRATION.md): early-fusion modality-dropout U-Net (E).

E = p9_fusion_dropout: the official DS-UNet UNet on VV + 13 optical bands stacked at the input (train_rrd.py
--arch unet_fusion), trained with B's mixture full=0.5, no_o2=0.25, no_opt=0.25. Unlike DS-UNet, the historical
optical image enters the same encoder as SAR, so it can condition the SAR features.

Reads experiments/robust/{arm}_seed{1..10}/metrics.json. Test F1 at each run's validation-selected threshold for the
condition (Step 6's rule). Paired over seeds 1-10; paired t-test, Holm within each family; Wilcoxon alongside.
    primary      E@no_o2 - E@no_opt (historical image within E);
                 interaction (E@no_o2 - E@no_opt) - (B@no_o2 - B@no_opt)                          (Holm over 2)
    secondary_rq1   E@no_o2 - SAR-only U-Net@full; E@no_o2 - B@no_o2                               (Holm over 2)
    secondary_full  E@full - input-fusion U-Net@full; E@full - A@full                              (Holm over 2)
Writes summary.json and STEP9_REPORT.md next to this file; refuses to overwrite.

Usage (repository root):  .venv/Scripts/python.exe experiments/robust/step9/analyze_step9.py [--runs DIR --out DIR]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analyze_phase3 import describe, fmt_p, holm, ms, paired  # noqa: E402
from src.evaluation.metrics import rates_from_counts  # noqa: E402

SEEDS = range(1, 11)
CONDITIONS = ("full", "no_o2", "no_opt", "o2_copy")
ARMS = {"p9_fusion_dropout": "E early-fusion U-Net, modality dropout", "p2_full": "A DS-UNet, full-only",
        "p2_dropout": "B DS-UNet, modality dropout", "p6_unet_sar": "SAR-only U-Net", "p6_unet_fusion": "input-fusion U-Net"}
METRICS = ("f1", "precision", "recall", "iou", "false_alarm_rate")
E_ARGS = {"arch": "unet_fusion", "sar": "hafner_vv", "train_conditions": "full=0.5,no_o2=0.25,no_opt=0.25",
          "steps": 2500, "eval_every": 25, "batch_size": 64, "crop_size": 32, "lr": 5e-5, "weight_decay": 5e-4}


def load(runs_dir: Path) -> dict:
    runs = {(a, s): json.loads((runs_dir / f"{a}_seed{s}" / "metrics.json").read_text(encoding="utf-8")) for a in ARMS for s in SEEDS}
    assert len({tuple(m["test_cities"]) for m in runs.values()}) == 1, "all runs must share the test cities"
    assert len({tuple(m["val_cities"]) for m in runs.values()}) == 1, "all runs must share the validation cities"
    for s in SEEDS:  # registered settings of E
        a = runs["p9_fusion_dropout", s]["args"]
        assert all(a[k] == v for k, v in E_ARGS.items()) and a["seed"] == s, f"E seed {s}: settings differ from registration"
    for m in runs.values():  # stored rates must follow from the stored counts
        for c in CONDITIONS:
            for thr in ("at_selected", "at_0.5"):
                r = m["test"][c][thr]
                assert (rates_from_counts({k: r[k] for k in ("tp", "fp", "fn", "tn")})["f1"] or 0.0) == (r["f1"] or 0.0)
    return runs


def family(pairs: dict, label: str | None = None) -> dict:
    rows = {k: paired(a, b) for k, (a, b) in pairs.items()}  # contrast is b - a
    for k, h in zip(rows, holm([r["t_p"] for r in rows.values()])):
        rows[k]["holm_p"] = h
    for k, h in zip(rows, holm([r["wilcoxon_p"] for r in rows.values()])):
        rows[k]["holm_wilcoxon_p"] = h
    if label:
        for r in rows.values():
            r["label"] = (f"{label} higher" if r["holm_p"] < 0.05 and r["mean_diff"] > 0 else
                          f"{label} lower" if r["holm_p"] < 0.05 and r["mean_diff"] < 0 else "no detectable difference")
    return rows


def analyse(runs: dict) -> dict:
    val = lambda arm, c, k="f1", thr="at_selected": [runs[arm, s]["test"][c][thr][k] or 0.0 for s in SEEDS]
    image = lambda arm: [p - q for p, q in zip(val(arm, "no_o2"), val(arm, "no_opt"))]  # historical image present - absent
    out = {"arms": {a: {thr: {c: {k: describe(val(a, c, k, thr)) for k in METRICS} for c in CONDITIONS}
                        for thr in ("at_selected", "at_0.5")} for a in ARMS},
           "selected_step": {a: [runs[a, s]["selected_step"] for s in SEEDS] for a in ARMS},
           "selected_threshold": {a: {c: [runs[a, s]["test"][c]["selected_threshold"] for s in SEEDS] for c in CONDITIONS} for a in ARMS}}
    out["primary"] = family({"E: image present - absent": (val("p9_fusion_dropout", "no_opt"), val("p9_fusion_dropout", "no_o2")),
                             "interaction: (E present - absent) - (B present - absent)": (image("p2_dropout"), image("p9_fusion_dropout"))},
                            "with the image / early fusion")
    out["secondary_rq1"] = family({"E@no_o2 - SAR-only": (val("p6_unet_sar", "full"), val("p9_fusion_dropout", "no_o2")),
                                   "E@no_o2 - B@no_o2": (val("p2_dropout", "no_o2"), val("p9_fusion_dropout", "no_o2"))}, "E")
    out["secondary_full"] = family({"E@full - input fusion": (val("p6_unet_fusion", "full"), val("p9_fusion_dropout", "full")),
                                    "E@full - A": (val("p2_full", "full"), val("p9_fusion_dropout", "full"))}, "E")
    out["descriptive_at_0.5"] = {"E: image present - absent": paired(val("p9_fusion_dropout", "no_opt", thr="at_0.5"),
                                                                    val("p9_fusion_dropout", "no_o2", thr="at_0.5"))}
    return out


def family_table(rows: dict, title: str) -> list[str]:
    L = [title, "", "| contrast | mean Δ | 95% CI | d_z | Δ > 0 | Holm p (t) | Holm p (Wilcoxon) | label |", "|---|---|---|---|---|---|---|---|"]
    for k, r in rows.items():
        L.append(f"| {k} | {r['mean_diff']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['dz']:+.2f} | {r['b_gt_a']}/10 | "
                 f"{fmt_p(r['holm_p'])} | {fmt_p(r['holm_wilcoxon_p'])} | {r.get('label', '')} |")
    return L + [""]


def report(out: dict) -> str:
    L = ["# Step 9: early-fusion modality-dropout U-Net (E)", "",
         "Pre-registered in `STEP9_PREREGISTRATION.md`. Generated by `analyze_step9.py`; numbers in `summary.json`.", ""]
    L += family_table(out["primary"], "## Primary: the historical image within E, and against B (Holm over 2)")
    L += family_table(out["secondary_rq1"], "## Secondary: E without the post-change image (Holm over 2)")
    L += family_table(out["secondary_full"], "## Secondary: E with full input (Holm over 2)")
    r = out["descriptive_at_0.5"]["E: image present - absent"]
    L += [f"Descriptive, threshold 0.5: E image present - absent {r['mean_diff']:+.3f} "
          f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}], {r['b_gt_a']}/10 seeds positive.", ""]
    for thr, name in (("at_selected", "validation-selected threshold"), ("at_0.5", "threshold 0.5")):
        L += [f"## Test metrics at the {name}, mean ± SD over 10 seeds", "",
              "| arm | condition | F1 | precision | recall | IoU | false-alarm rate |", "|---|---|---|---|---|---|---|"]
        for arm, name_ in ARMS.items():
            for c in CONDITIONS:
                e = out["arms"][arm][thr][c]
                L.append(f"| {name_} | {c} | " + " | ".join(ms(e[k]) for k in METRICS) + " |")
        L.append("")
    return "\n".join(L) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=ROOT / "experiments" / "robust")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = p.parse_args()
    if (args.out / "summary.json").exists():
        raise SystemExit(f"{args.out / 'summary.json'} exists; refusing to overwrite")
    out = analyse(load(args.runs))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (args.out / "STEP9_REPORT.md").write_text(report(out), encoding="utf-8")
    print(f"wrote {args.out / 'summary.json'} and STEP9_REPORT.md")


if __name__ == "__main__":
    main()
