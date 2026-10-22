"""Fill experiments/robust/PHASE4B_REPORT.template.md -> PHASE4B_REPORT.md from phase4b/summary.json and p4b_audit.json.

Generated parts: integrity facts, the primary table (with pre-specified labels), the clean-regression flag, selected
steps, and the analysis tables pasted unchanged. Interpretation text lives in the template and must quote only numbers
that are checked here: every {KEY} placeholder is computed from summary.json.

Usage:
    .venv/Scripts/python.exe build_phase4b_report.py
"""

import json
from pathlib import Path

R = Path(__file__).resolve().parent / "experiments" / "robust"
OUT = R / "phase4b"
PRIMARY = ("clean", "z10", "z30", "z60", "c30__erode4", "c30__dilate4", "c30__miss25")


def main() -> None:
    s = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    B, C = s["arms"]["p2_dropout"], s["arms"]["p4b_spatial"]
    rows = ["| condition | B F1 | C F1 | mean Δ (C − B) | 95% CI | d_z | C > B | Holm p (t) | Holm p (Wilcoxon) | label |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for c in PRIMARY:
        r = s["primary"][c]
        rows.append(f"| {c} | {B[c]['mean']:.3f} ± {B[c]['sd']:.3f} | {C[c]['mean']:.3f} ± {C[c]['sd']:.3f} | {r['mean_diff']:+.3f} | "
                    f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['dz']:+.2f} | {r['b_gt_a']}/10 | {r['holm_p']:.1e} | "
                    f"{r['holm_wilcoxon_p']:.4f} | {r['label']} |")
    au_path = OUT / "p4b_audit.json"
    if au_path.exists():
        au = json.loads(au_path.read_text(encoding="utf-8"))
        only_report = au["failed"] and all("REPORT.md exists" in f["check"] for f in au["failed"])
        audit = f"**{au['n_checks']} checks, {au['n_failed']} failed**" + (" (only the report-table check; report not yet written)" if only_report else "")
    else:
        audit = "not yet run"
    sec = s["secondary_other_conditions"]
    p2 = s["secondary_phase2_conditions"]
    fill = {
        "PRIMARY_TABLE": "\n".join(rows), "AUDIT": audit,
        "CLEAN_REGRESSION": "yes" if s["substantial_clean_regression"] else "no",
        "STEPS_B": str(s["arms"]["p2_dropout"]["selected_steps"]), "STEPS_C": str(s["arms"]["p4b_spatial"]["selected_steps"]),
        "N_SEC_BETTER": str(sum(r["holm_p"] < 0.05 and r["mean_diff"] > 0 for r in sec.values())),
        "N_SEC_WORSE": str(sum(r["holm_p"] < 0.05 and r["mean_diff"] < 0 for r in sec.values())), "N_SEC": str(len(sec)),
        "P2_ROWS": "; ".join(f"{c} {r['mean_diff']:+.3f} (Holm p {r['holm_p']:.2g})" for c, r in p2.items()),
        "TABLES": (OUT / "tables.md").read_text(encoding="utf-8"),
    }
    for c in ("c10", "c30", "c60", "z30", "clean", "c30__erode4", "c30__miss25", "c30__dilate4", "c30__erode8", "c30__miss50"):
        fill[f"B_{c}"], fill[f"C_{c}"] = f"{B[c]['mean']:.3f}", f"{C[c]['mean']:.3f}"
    for c in ("c10", "c30", "c60", "c30__erode8", "c30__miss50", "c30__erode2", "c30__miss10", "c60__erode8"):
        if c in sec:
            fill[f"D_{c}"] = f"{sec[c]['mean_diff']:+.3f}"
            fill[f"H_{c}"] = f"{sec[c]['holm_p']:.2g}"
    io = s["inside_outside"]
    for arm, k in (("p2_dropout", "B"), ("p4b_spatial", "C")):
        for c in ("c30", "z30", "c30__erode4", "c30__miss25", "c30__dilate4"):
            for m in ("inside_false_alarm_rate", "outside_false_alarm_rate", "inside_recall", "outside_f1"):
                fill[f"IO_{k}_{c}_{m}"] = f"{io[arm][c][m]['mean']:.3f}"
    # Exploratory pattern over all 34 conditions (primary + secondary; not independent, no test): conditions where
    # cloud-valued pixels reach the model (no fill, erosion, missed clouds) vs conditions with only zero-filled clouds
    # (clean, oracle, dilation).
    every = {**s["primary"], **sec}
    cloud_valued = [c for c in every if c != "clean" and c[0] == "c" and ("__" not in c or "__erode" in c or "__miss" in c)]
    zero_only = [c for c in every if c not in cloud_valued]
    fill["N_CV"], fill["N_ZO"] = str(len(cloud_valued)), str(len(zero_only))
    fill["N_CV_NEG"] = str(sum(every[c]["mean_diff"] < 0 for c in cloud_valued))
    fill["N_ZO_POS"] = str(sum(every[c]["mean_diff"] > 0 for c in zero_only))
    fill["CV_RANGE"] = f"{min(every[c]['mean_diff'] for c in cloud_valued):+.3f} to {max(every[c]['mean_diff'] for c in cloud_valued):+.3f}"
    fill["ZO_RANGE"] = f"{min(every[c]['mean_diff'] for c in zero_only):+.3f} to {max(every[c]['mean_diff'] for c in zero_only):+.3f}"
    met = s["metrics"]
    for arm, k in (("p2_dropout", "B"), ("p4b_spatial", "C")):
        for m in ("precision", "recall", "false_alarm_rate"):
            fill[f"M_{k}_clean_{m}"] = f"{met[arm]['clean'][m]['mean']:.3f} ± {met[arm]['clean'][m]['sd']:.3f}"
    runs = {arm: [json.loads((R / d / "runs" / f"{arm}_seed{i}.json").read_text(encoding="utf-8")) for i in range(1, 11)]
            for arm, d in (("p2_dropout", "phase4abis"), ("p4b_spatial", "phase4b"))}
    fill["TAU_B"] = str([r["tau_clean"] for r in runs["p2_dropout"]])
    fill["TAU_C"] = str([r["tau_clean"] for r in runs["p4b_spatial"]])
    fill["N_ANCHOR_RUNS"] = str(sum(all(r["anchors"].values()) for r in runs["p4b_spatial"]))
    secs = [json.loads((R / f"p4b_spatial_seed{i}" / "metrics.json").read_text(encoding="utf-8"))["train_seconds"] for i in range(1, 11)]
    fill["TRAIN_SECONDS"] = ", ".join(f"{x:.0f}" for x in secs)

    # Claims the template's prose makes; the build fails if the results do not support them.
    claims = {
        "all 7 primary labels are 'no detectable difference'": all(r["label"] == "no detectable difference" for r in s["primary"].values()),
        "no substantial clean regression": not s["substantial_clean_regression"],
        "no secondary condition has Holm p < 0.05": all(r["holm_p"] >= 0.05 for r in list(sec.values()) + list(p2.values())),
        "smallest primary Holm p > 0.5": min(r["holm_p"] for r in s["primary"].values()) > 0.5,
        "most cloud-valued conditions have C < B": sum(every[c]["mean_diff"] < 0 for c in cloud_valued) >= 0.75 * len(cloud_valued),
        "most zero-only conditions have C > B": sum(every[c]["mean_diff"] > 0 for c in zero_only) >= 0.75 * len(zero_only),
        "every primary |mean Δ| < 0.02": all(abs(r["mean_diff"]) < 0.02 for r in s["primary"].values()),
        "C flags more of the unmasked cloud as change than B (FAR in M, c30)": io["p4b_spatial"]["c30"]["inside_false_alarm_rate"]["mean"] > io["p2_dropout"]["c30"]["inside_false_alarm_rate"]["mean"],
        "C has larger seed SD of clean precision and recall than B": all(met["p4b_spatial"]["clean"][m]["sd"] > met["p2_dropout"]["clean"][m]["sd"] for m in ("precision", "recall")),
        "mean clean Δ is positive": s["primary"]["clean"]["mean_diff"] > 0,
        "oracle Δ positive at z10, z30, z60": all(s["primary"][c]["mean_diff"] > 0 for c in ("z10", "z30", "z60")),
        "imperfect-mask primary Δ: erode4 < 0, miss25 < 0, dilate4 > 0": s["primary"]["c30__erode4"]["mean_diff"] < 0 and s["primary"]["c30__miss25"]["mean_diff"] < 0 and s["primary"]["c30__dilate4"]["mean_diff"] > 0,
        "C is lower than B on no_opt and o2_copy (Phase 2 conditions)": p2["no_opt"]["mean_diff"] < 0 and p2["o2_copy"]["mean_diff"] < 0,
        "all 10 C evaluation runs passed every anchor": fill["N_ANCHOR_RUNS"] == "10",
    }
    bad = [k for k, ok in claims.items() if not ok]
    if bad:
        raise SystemExit(f"prose claims not supported by summary.json: {bad}")
    text = (R / "PHASE4B_REPORT.template.md").read_text(encoding="utf-8")
    for k, v in fill.items():
        text = text.replace("{" + k + "}", v)
    import re
    left = re.findall(r"\{[A-Z][A-Za-z0-9_]+\}", text.split("## 9.")[0])
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    (R / "PHASE4B_REPORT.md").write_text(text, encoding="utf-8")
    print("PHASE4B_REPORT.md written")


if __name__ == "__main__":
    main()
