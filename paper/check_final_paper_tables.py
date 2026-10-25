"""Check every cell of the results tables in paper/FINAL_RESEARCH_PAPER.md against the audited summary JSONs.

§5.1-5.7 reuse the robustness draft's checks (check_robustness_paper_tables.py). §4.1-4.5 are checked against
experiments/robust/phase{6,7,8}/summary.json and Appendix A against experiments/paper_statistics.json, with the same
tolerances: 3 decimals to 0.0005, 2 decimals to 0.005, p-values to 6% relative. Exits 1 on any mismatch.

Usage:
    .venv/Scripts/python.exe paper/check_final_paper_tables.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_robustness_paper_tables as base  # noqa: E402
from check_robustness_paper_tables import check, load, ms, table  # noqa: E402

METRICS = ["f1", "precision", "recall", "iou", "false_alarm_rate"]


def contrast(label, cells, r):
    """mean Δ, 95% CI, d_z, Δ > 0, Holm p and label cells of one generated-contrast row."""
    check(f"{label} Δ", cells[0], [r["mean_diff"]]), check(f"{label} CI", cells[1], r["ci95"])
    check(f"{label} dz", cells[2], [r["dz"]]), check(f"{label} Δ>0", cells[3], f"{r['b_gt_a']}/10", "count")
    check(f"{label} Holm", cells[4], [r["holm_p"]], "p"), check(f"{label} label", cells[5], r["label"].replace(" ", ""), "count")


def main() -> None:
    text = (base.ROOT / "paper" / "FINAL_RESEARCH_PAPER.md").read_text(encoding="utf-8")
    base.check_robustness_tables(text, {f"4.{i}": f"### 5.{i}" for i in range(1, 8)})

    # 4.1 RQ4 (Step 6): metrics table, then contrasts
    s6 = load("phase6/summary.json")
    arm6 = {"DS-UNet A": "p2_full", "DS-UNet B (modality dropout)": "p2_dropout", "SAR-only U-Net": "p6_unet_sar",
            "optical-only U-Net": "p6_unet_opt", "input-fusion U-Net": "p6_unet_fusion"}
    for row in table(text, "### 4.1"):
        for i, k in enumerate(METRICS, 1):
            check(f"4.1 {row[0]} {k}", row[i], ms(s6["arms"][arm6[row[0]]]["at_selected"]["full"][k]))
    for row in table(text.split("### 4.1", 1)[1].split("| input-fusion U-Net |", 1)[1], "\n"):  # the table after
        contrast(f"4.1 {row[0]}", row[1:], s6["primary_rq4"][row[0].replace("−", "-")])

    # 4.2 RQ1 (Step 6)
    no_o2 = {"B@no_o2 − SAR-only": "p2_dropout", "async@no_o2 − SAR-only": "p6_async"}
    for row in table(text, "### 4.2"):
        check(f"4.2 {row[0]} F1", row[1], ms(s6["arms"][no_o2[row[0]]]["at_selected"]["no_o2"]["f1"]))
        contrast(f"4.2 {row[0]}", row[2:], s6["primary_rq1"][row[0].replace("−", "-")])

    # 4.3 RQ2 (Step 8): F1 per tag
    s8 = load("phase8/summary.json")
    tags = ["oscd", "gee_t1", "gap01", "gap03", "gap06", "gap12", "gap18"]
    for row in table(text, "### 4.3"):
        arm = {"B dropout": "p2_dropout", "async": "p6_async"}[row[0]]
        for i, t in enumerate(tags, 1):
            check(f"4.3 {arm} {t}", row[i], ms(s8["arms"][arm][t]["f1"]))

    # 4.4 RQ3 (Step 7): metrics table, then contrasts
    s7 = load("phase7/summary.json")
    arm7 = {"SAR-only U-Net, VV": "p6_unet_sar", "SAR-only U-Net, VH": "p7_unet_vh", "SAR-only U-Net, VV+VH": "p7_unet_vvvh",
            "DS-UNet, VV": "p2_full", "DS-UNet, VH": "p7_dsunet_vh", "DS-UNet, VV+VH": "p7_dsunet_vvvh"}
    for row in table(text, "### 4.4"):
        for i, k in enumerate(METRICS, 1):
            check(f"4.4 {row[0]} {k}", row[i], ms(s7["arms"][arm7[row[0]]]["9_cities"][k]))
    for row in table(text.split("### 4.4", 1)[1].split("| DS-UNet, VV+VH |", 1)[1], "\n"):
        fam, c = row[0].split(": ")
        contrast(f"4.4 {row[0]}", row[1:], s7["primary_rq3" if fam == "SAR-only" else "secondary_dsunet"][c.replace("−", "-")])

    # 4.5 RQ5 (Step 8): false-alarm rate by WorldCover class
    for row in table(text, "### 4.5"):
        check(f"4.5 {row[0]} share", row[1], [100 * s8["worldcover_share"][row[0]]])
        r = s8["rq5_landcover"][f"B@oscd - SAR-only, {row[0]}"]
        check(f"4.5 {row[0]} Δ", row[2], [r["mean_diff"]]), check(f"4.5 {row[0]} CI", row[3], r["ci95"])
        check(f"4.5 {row[0]} Δ>0", row[4], f"{r['b_gt_a']}/10", "count"), check(f"4.5 {row[0]} Holm", row[5], [r["holm_p"]], "p")
        check(f"4.5 {row[0]} label", row[6], r["label"].replace(" ", ""), "count")

    # Appendix A (preliminary study)
    ps = json.loads((base.ROOT / "experiments" / "paper_statistics.json").read_text(encoding="utf-8"))
    names = {"SAR-only U-Net": ("models", "sar_baseline"), "Optical-only U-Net": ("models", "optical_baseline"),
             "Early Fusion U-Net": ("models", "fusion_baseline"), "Sync Dual-Stream CNN": ("models", "dual_stream_baseline"),
             "Naive Missing-Optical CNN": ("models", "asynchronous_dual_stream"),
             "Clean Missing-Optical CNN": ("models", "asynchronous_dual_stream_clean"),
             "Transformer Self-Attn (original)": ("models", "transformer_self_attention"),
             "Transformer Cross-Attn (original)": ("models", "transformer_cross_attention"),
             "Transformer Cross-Attn (pw=1)": ("models", "transformer_cross_attention_pw1"),
             "Transformer Self-Attn (tuned, wd=1e-4)": ("tuned_transformers", "self"),
             "Transformer Cross-Attn (tuned, wd=1e-4)": ("tuned_transformers", "cross")}
    for row in table(text, "**Table A1"):
        m = ps[names[row[0]][0]][names[row[0]][1]]
        check(f"A1 {row[0]} n", row[2], str(m["n"]), "count")
        for i, k in enumerate(("f1", "precision", "recall", "iou", "far"), 3):
            check(f"A1 {row[0]} {k}", row[i], [m[f"final_test_{k}_mean"], m[f"final_test_{k}_sd"]])
    tests = dict(zip(("Sync dual-stream − Clean missing-optical", "Clean missing-optical − SAR-only", "Sync dual-stream − SAR-only",
                      "Sync dual-stream − Early fusion", "Clean − Naive missing-optical", "Transformer Cross-Attn pw=41 − pw=1"),
                     ps["tests"].values()))
    for row in table(text, "**Table A2"):
        t = tests[row[0]]
        check(f"A2 {row[0]} Δ", row[1], [t["mean_diff"]]), check(f"A2 {row[0]} n", row[2], f"{t['n_a']}vs{t['n_b']}", "count")
        check(f"A2 {row[0]} p", row[3], [t["welch_p"]], "p"), check(f"A2 {row[0]} Holm", row[4], [t["welch_p_holm"]], "p")

    print(f"{base.n_checks} table cells checked, {len(base.fails)} mismatches")
    for f in base.fails:
        print("  MISMATCH", f)
    sys.exit(1 if base.fails else 0)


if __name__ == "__main__":
    main()
