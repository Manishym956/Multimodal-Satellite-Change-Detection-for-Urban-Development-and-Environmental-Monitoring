"""Check every cell of the results tables in paper/final_submission_v2.md against the audited summary JSONs.

§5.1-5.7 reuse the robustness draft's checks (check_robustness_paper_tables.py). §4.1-4.5 are checked against
experiments/robust/phase{6,7,8}/summary.json and Appendix A against experiments/paper_statistics.json, with the same
tolerances: 3 decimals to 0.0005, 2 decimals to 0.005, p-values to 6% relative. Exits 1 on any mismatch.

Usage:
    .venv/Scripts/python.exe paper/check_final_submission_v2_tables.py
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


def labelled(label, cells, r, p_key="holm_p", has_dz=True):
    """Revision tables: Δ, CI, (d_z), Δ > 0, p; the label must follow from sign and p (< 0.05) or say not detectable."""
    i = 0
    check(f"{label} Δ", cells[i], [r["mean_diff"]]), check(f"{label} CI", cells[i + 1], r["ci95"])
    i += 2
    if has_dz:
        check(f"{label} dz", cells[i], [r["dz"]])
        i += 1
    check(f"{label} Δ>0", cells[i], f"{r['b_gt_a']}/10", "count"), check(f"{label} p", cells[i + 1], [r[p_key]], "p")
    lab = cells[i + 2]
    if lab.startswith("("):  # "(descriptive)" rows carry no label
        return
    want_null = r[p_key] >= 0.05
    ok = (lab == "no detectable difference") == want_null and (
        want_null or (("lower" in lab) == (r["mean_diff"] < 0) and ("higher" in lab) == (r["mean_diff"] > 0)))
    base.n_checks += 1
    if not ok:
        base.fails.append(f"{label} label: '{lab}' vs Δ {r['mean_diff']:+.3f}, p {r[p_key]:.3g}")


def check_revision_tables(text) -> None:
    s4b, s9, s10, s11 = (load(p) for p in ("stage4b/stage4b_summary.json", "step9/summary.json", "step10/summary.json",
                                           "realcloud/summary.json"))
    dec = s4b["descriptive"]["decomposition_tau_clean"]
    t4 = {"B: image present − absent": s4b["primary"], "B, image present − SAR-only": dec["present - SAR-only"],
          "B, image absent − SAR-only": dec["absent - SAR-only"]}
    for row in table(text, "**Table 6."):
        labelled(f"T4 {row[0]}", row[1:], t4[row[0]], "t_p")
    t5 = {"E: image present − absent": s9["primary"]["E: image present - absent"],
          "(E present − absent) − (B present − absent)": s9["primary"]["interaction: (E present - absent) - (B present - absent)"],
          "E@no_o2 − SAR-only": s9["secondary_rq1"]["E@no_o2 - SAR-only"], "E@no_o2 − B@no_o2": s9["secondary_rq1"]["E@no_o2 - B@no_o2"],
          "E@full − input fusion": s9["secondary_full"]["E@full - input fusion"], "E@full − A": s9["secondary_full"]["E@full - A"]}
    for row in table(text, "**Table 7."):
        labelled(f"T5 {row[0]}", row[1:], t5[row[0]])
    t6 = {"E@no_opt − SAR-only": s10["primary"]["E@no_opt - SAR-only"],
          "E: image present − absent": s10["secondary"]["E: image present - absent (Step 9 H1)"],
          "E@no_opt − SAR-only, threshold 0.5": s10["secondary"]["E@no_opt - SAR-only, threshold 0.5"]}
    for row in table(text, "**Table 8."):
        labelled(f"T6 {row[0]}", row[1:], t6[row[0]])
    t7 = {"B: rb95 mask − no mask": s11["primary"]["B real_rb95 - B real"],
          "B with rb95 mask − SAR-only": s11["primary"]["B real_rb95 - SAR-only"],
          "B − A, no mask": s11["primary"]["B real - A real"], "B: bal mask − no mask": s11["secondary"]["B real_bal - B real"],
          "B: rb95 mask − image discarded": s11["secondary"]["B real_rb95 - B missing"],
          "A, no mask − SAR-only": s11["secondary"]["A real - SAR-only"],
          "B: Earth Engine control − OSCD image": s11["secondary"]["B control - B oscd"]}
    for row in table(text, "**Table 18."):
        labelled(f"T7 {row[0]}", row[1:], t7[row[0]])
    cond = {"OSCD image (clear)": "oscd", "Earth Engine control (clear, same day)": "control",
            "real cloudy scene, no mask": "real", "real cloudy scene, bal mask": "real_bal",
            "real cloudy scene, rb95 mask": "real_rb95", "image discarded": "missing"}
    for row in table(text, "**Table 19."):
        for cell, arm in zip(row[1:], ("p2_full", "p2_dropout"), strict=True):
            check(f"T8 {row[0]} {arm}", cell, ms(s11["arms"]["cloudy"][arm][cond[row[0]]]["f1"]))


def main() -> None:
    text = (base.ROOT / "paper" / "final_submission_v2.md").read_text(encoding="utf-8")
    base.check_robustness_tables(text, {"4.1": "### 5.1", "4.2": "### 5.2", "4.3": "### 5.3", "4.4": "**Table E1",
                                        "4.5": "**Table E2", "4.6": "**Table E3", "4.7": "**Table E4"})
    check_revision_tables(text)

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
    for row in table(text, "### 4.4"):
        arm = {"B dropout": "p2_dropout", "async": "p6_async"}[row[0]]
        for i, t in enumerate(tags, 1):
            check(f"4.3 {arm} {t}", row[i], ms(s8["arms"][arm][t]["f1"]))

    # 4.4 RQ3 (Step 7): metrics table, then contrasts
    s7 = load("phase7/summary.json")
    arm7 = {"SAR-only U-Net, VV": "p6_unet_sar", "SAR-only U-Net, VH": "p7_unet_vh", "SAR-only U-Net, VV+VH": "p7_unet_vvvh",
            "DS-UNet, VV": "p2_full", "DS-UNet, VH": "p7_dsunet_vh", "DS-UNet, VV+VH": "p7_dsunet_vvvh"}
    for row in table(text, "### 4.5"):
        for i, k in enumerate(METRICS, 1):
            check(f"4.4 {row[0]} {k}", row[i], ms(s7["arms"][arm7[row[0]]]["9_cities"][k]))
    for row in table(text.split("### 4.5", 1)[1].split("| DS-UNet, VV+VH |", 1)[1], "\n"):
        fam, c = row[0].split(": ")
        contrast(f"4.4 {row[0]}", row[1:], s7["primary_rq3" if fam == "SAR-only" else "secondary_dsunet"][c.replace("−", "-")])

    # 4.5 RQ5 (Step 8): false-alarm rate by WorldCover class
    for row in table(text, "### 4.6"):
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
    # Tables A2 and A3 were removed from the v2 manuscript (Appendix A shortened).

    # 4.6 and 5.8: exploratory revision tables (Stage 4), against experiments/robust/stage4/stage4_summary.json
    s4 = load("stage4/stage4_summary.json")
    for row, r in zip(table(text, "### 4.7"), s4["E2_city_vs_seed"], strict=True):
        n = r["n_cities"]
        check(f"4.6 {row[0]} cities", row[1], str(n), "count"), check(f"4.6 {row[0]} Δ", row[2], [r["seed_mean_diff"]])
        check(f"4.6 {row[0]} seed CI", row[3], r["seed_ci95"]), check(f"4.6 {row[0]} SE seeds", row[4], [r["se_seeds"]])
        check(f"4.6 {row[0]} SE cities", row[5], [r["se_cities_jackknife"]])
        check(f"4.6 {row[0]} sign", row[6], f"{r['cities_same_sign']}/{n}", "count")
        check(f"4.6 {row[0]} boot CI", row[7], r["boot_ci95"])
        check(f"4.6 {row[0]} LOCO", row[8], [r["loco_min"], r["loco_max"]])
    e5 = {x["contrast"]: x for x in s4["E5_fallback_rule"]}
    for row in table(text, "**Table E5"):
        for cell, m in zip(row[1:], ("cloud", "bal", "rb95", "rb99", "oracle"), strict=True):
            x = e5[f"composite clouds {row[0]}: B with mask {m} - SAR-only"]
            check(f"5.8 {row[0]} {m}", cell, [x["mean_diff"], *x["ci95"]])

    # v2 Table 14: per-city differences (Stage 4 E2 and Step 10 city level)
    e2 = s4["E2_city_vs_seed"]
    c10 = load("step10/summary.json")["city_level"]
    for row in table(text, "**Table 14."):
        if row[0].startswith("pooled, 10"):
            for cell, x in zip(row[2:], (e2[0]["seed_mean_diff"], e2[1]["seed_mean_diff"], e2[2]["seed_mean_diff"], c10["delta"])):
                check(f"T14 {row[0]}", cell, [x])
            continue
        if row[0].startswith("pooled, without"):
            lv = [next(d["mean_diff"] for d in e["loco"] if d["dropped"] == "lasvegas") for e in e2[:3]]
            for cell, x in zip(row[2:], lv + [c10["left_out"]["lasvegas"]]):
                check(f"T14 {row[0]}", cell, [x])
            continue
        city = row[0]
        check(f"T14 {city} share", row[1], [100 * e2[0]["change_pixel_share"][city]])
        for k, cell in enumerate(row[2:5]):
            check(f"T14 {city} c{k}", cell, [e2[k]["per_city_mean_diff"][city]])
        check(f"T14 {city} step10", row[5], [c10["per_city_delta"][city]])

    # v2 Table 20: Step 11 overcast and all-city families
    s11 = load("realcloud/summary.json")
    keys = {"B: rb95 mask − no mask": "B real_rb95 - B real", "B with rb95 mask − SAR-only": "B real_rb95 - SAR-only",
            "B − A, no mask": "B real - A real"}
    for row in table(text, "**Table 20."):
        r = s11[{"3 overcast": "overcast", "all 10": "all"}[row[0]]][keys[row[1]]]
        check(f"T20 {row[0]} {row[1]} Δ", row[2], [r["mean_diff"]]), check(f"T20 {row[0]} {row[1]} CI", row[3], r["ci95"])
        check(f"T20 {row[0]} {row[1]} Δ>0", row[4], f"{r['b_gt_a']}/10", "count")
        check(f"T20 {row[0]} {row[1]} Holm", row[5], [r["holm_p"]], "p")

    print(f"{base.n_checks} table cells checked, {len(base.fails)} mismatches")
    for f in base.fails:
        print("  MISMATCH", f)
    sys.exit(1 if base.fails else 0)


if __name__ == "__main__":
    main()
