"""Fill experiments/robust/PHASE4ABIS_REPORT.template.md -> PHASE4ABIS_REPORT.md and check every claim made in its prose.

Numbers are formatted from phase4abis/summary.json (registered analysis); qualitative claims are asserted against it.
The addendum section is filled from phase4abis/addendum_inside_outside/summary.json when it exists.

Usage:
    .venv/Scripts/python.exe build_phase4abis_report.py
"""

import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent / "experiments" / "robust"
OUT = R / "phase4abis"
VARIANTS = ("erode2", "erode4", "erode8", "dilate2", "dilate4", "dilate8", "miss10", "miss25", "miss50")
DIL, UNDER = VARIANTS[3:6], VARIANTS[:3] + VARIANTS[6:]


def main() -> None:
    s = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    A, B = s["arms"]["p2_full"], s["arms"]["p2_dropout"]
    fam = s["families"]
    prim = fam["primary_B_c30_imperfect_minus_oracle"]
    f1 = lambda arm, c, thr="clean": arm[thr][c]["f1"]["mean"]
    cost = lambda rows, vs: "/".join(f"{rows[v]['mean_diff']:+.3f}" for v in vs)
    es = s["error_stats"]
    reg = s["regions"]

    checks = {
        "all 9 primary Holm p < 0.05, 10/10 below oracle": all(r["holm_p"] < 0.05 and r["b_gt_a"] == 0 for r in prim.values()),
        "B dilation retained >= 0.97 at c30": all(B["clean"][f"c30__{v}"]["retained_gain"]["mean"] >= 0.965 for v in DIL),
        "'most' label exactly erode2@c10, miss10@c10, erode2@c60 among under-masking": sorted(
            c for c in B["clean"] if "__" in c and not c.split("__")[1].startswith("dilate") and B["clean"][c]["retained_gain"]["most_retained"])
            == sorted(["c10__erode2", "c10__miss10", "c60__erode2"]),
        "B every imperfect > cloud at c30": min(f1(B, f"c30__{v}") for v in VARIANTS) > f1(B, "c30"),
        "worst c30 retained about a quarter": all(0.2 <= B["clean"][f"c30__{v}"]["retained_gain"]["mean"] <= 0.3 for v in ("erode8", "miss50")),
        "B under-mask max cost (c10, c60) is miss50@c60": min((r["mean_diff"], f"{lv}{v}") for lv in ("c10", "c60")
            for v, r in fam[f"secondary_p2_dropout_{lv}_imperfect_minus_oracle"].items())[1] == "c60miss50",
        "B - A > 0 in 10/10 at c30 for all": all(r["b_gt_a"] == 10 for r in fam["secondary_B_minus_A_c30"].values()),
        "A dilation cost > B dilation cost at c30": all(fam["secondary_p2_full_c30_imperfect_minus_oracle"][v]["mean_diff"] < prim[v]["mean_diff"] for v in DIL),
        "erode2 0.116 -> -0.100, miss10 0.100 -> -0.090": (f"{es['c30__erode2']['missed_cloud_frac']:.3f}", f"{prim['erode2']['mean_diff']:+.3f}",
            f"{prim['miss10']['mean_diff']:+.3f}") == ("0.116", "-0.100", "-0.090"),
    }
    under_other = {(thr, v): f1(B, "z30", thr) - f1(B, f"c30__{v}", thr) for thr in ("0.5", "level") for v in UNDER}
    for thr in ("0.5", "level"):
        checks[f"threshold {thr}: dilation < 0.02"] = all(f1(B, "z30", thr) - f1(B, f"c30__{v}", thr) < 0.02 for v in DIL)
    checks["smallest under-masking cost at other thresholds is miss10 at 0.5"] = min(under_other, key=under_other.get) == ("0.5", "miss10")
    b_missed = [reg["p2_dropout"][f"{lv}__{v}"]["missed_input_false_alarm_rate"]["mean"] for lv in ("c10", "c30", "c60")
                for v in ("erode4", "erode8", "miss10", "miss25", "miss50")]
    b_missed_clean = [reg["p2_dropout"][f"{lv}__{v}"]["missed_clean_false_alarm_rate"]["mean"] for lv in ("c10", "c30", "c60")
                      for v in ("erode4", "erode8", "miss10", "miss25", "miss50")]
    checks["B missed-cloud FAR with clean input about 0.04"] = all(0.025 <= x <= 0.055 for x in b_missed_clean)
    e2 = [reg["p2_dropout"][f"{lv}__erode2"]["missed_input_false_alarm_rate"]["mean"] for lv in ("c10", "c30", "c60")]
    over = lambda arm, key: [reg[arm][f"c30__{v}"][key]["mean"] for v in DIL]
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise SystemExit(f"prose claims not supported by summary.json: {failed}")

    rng = lambda xs: f"{min(xs):.3f}–{max(xs):.3f}"
    rows = ["| mask | missed cloud | over-masked clear | B F1 | mean Δ | 95% CI | d_z | imperfect > oracle | Holm p (t) | Holm p (Wilcoxon) | retained gain |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        r, c, e = prim[v], B["clean"][f"c30__{v}"], es[f"c30__{v}"]
        rows.append(f"| {v} | {e['missed_cloud_frac']:.3f} | {e['overmasked_clear_frac']:.3f} | {c['f1']['mean']:.3f} ± {c['f1']['sd']:.3f} | "
                    f"{r['mean_diff']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['dz']:+.2f} | {r['b_gt_a']}/10 | "
                    f"{r['holm_p']:.1e} | {r['holm_wilcoxon_p']:.4f} | {c['retained_gain']['mean']:.2f}{' (most)' if c['retained_gain']['most_retained'] else ''} |")

    audit_path = OUT / "p4abis_audit.json"
    if audit_path.exists():
        au = json.loads(audit_path.read_text(encoding="utf-8"))
        nonreport = [c for c in au["failed"] if "REPORT.md exists" not in c["check"]]
        audit = (f"Last audit run: **{au['n_checks']} checks, {au['n_failed']} failed**"
                 + (" (only the report-table check, because the report did not exist yet; re-run after this build)." if au["failed"] and not nonreport else ".")
                 + " (`phase4abis/p4abis_audit.json`).")
    else:
        audit = "Audit not yet run."

    add_path = OUT / "addendum_inside_outside" / "summary.json"
    if add_path.exists():
        ad = json.loads(add_path.read_text(encoding="utf-8"))
        io, iou = ad["inside_outside"], ad["iou"]
        g = lambda arm, c, k: io[arm][c][k]["mean"]
        notes = [
            "| arm | 30% condition | F1 in M | recall in M | FAR in M | F1 outside M | FAR outside M | pooled IoU |",
            "|---|---|---|---|---|---|---|---|"]
        for arm, name in (("p2_full", "A"), ("p2_dropout", "B")):
            for c, label in (("clean@c30", "clean"), ("c30", "cloud, no fill"), ("z30", "oracle fill"), *((f"c30__{v}", v) for v in ("erode4", "dilate4", "miss25"))):
                pooled = iou[arm]["clean" if c.startswith("clean@") else c]["mean"]
                notes.append(f"| {name} | {label} | {g(arm, c, 'inside_f1'):.3f} | {g(arm, c, 'inside_recall'):.3f} | {g(arm, c, 'inside_false_alarm_rate'):.3f} | "
                             f"{g(arm, c, 'outside_f1'):.3f} | {g(arm, c, 'outside_false_alarm_rate'):.3f} | {pooled:.3f} |")
        notes += ["", "Means over 10 seeds at the frozen clean threshold. 'clean' = clean input scored on the same pixel partition."]
        runs = list((OUT / "addendum_inside_outside" / "runs").glob("*.json"))
        ok = len(runs) == 20 and all(all(json.loads(p.read_text(encoding="utf-8"))["anchors"].values()) for p in runs)
        anchor = f"Anchor result: {len(runs)} runs, all anchors {'passed' if ok else 'NOT passed'}."
        addendum = "\n".join(notes)
        far_out = {c: g("p2_dropout", c, "outside_false_alarm_rate") for c in ("c30", "z30", "c30__erode4", "c30__miss25", "c30__dilate4")}
        f1_out = {c: g("p2_dropout", c, "outside_f1") for c in ("c30", "z30", "c30__erode4", "c30__dilate4")}
        a_in = {c: g("p2_full", c, "inside_false_alarm_rate") for c in ("c30", "z30")}
        claims = {
            "B erode4 outside-M FAR within 0.005 of no fill, > oracle + 0.02": abs(far_out["c30__erode4"] - far_out["c30"]) < 0.005
                and far_out["c30__erode4"] > far_out["z30"] + 0.02,
            "B miss25 outside FAR between oracle and no fill": far_out["z30"] < far_out["c30__miss25"] < far_out["c30"],
            "B dilate4 outside FAR within 0.005 of oracle": abs(far_out["c30__dilate4"] - far_out["z30"]) < 0.005,
            "B dilate4 outside F1 below oracle": f1_out["c30__dilate4"] < f1_out["z30"],
            "A inside-M FAR oracle about 0.5": 0.45 <= a_in["z30"] <= 0.55,
        }
        bad = [k for k, ok in claims.items() if not ok]
        if bad:
            raise SystemExit(f"addendum claims not supported: {bad}")
        obs = [
            f"- **Unmasked cloud edges drive the spill-over outside the cloud.** For B at 30% with erode4, the false-alarm rate on clear "
            f"pixels outside M is {far_out['c30__erode4']:.3f}, the same as with no fill at all ({far_out['c30']:.3f}); the oracle gives "
            f"{far_out['z30']:.3f}. With miss25 it is {far_out['c30__miss25']:.3f}. Outside-M F1: {f1_out['c30__erode4']:.3f} (erode4), "
            f"{f1_out['c30']:.3f} (no fill), {f1_out['z30']:.3f} (oracle). A thin rim of cloud-valued pixels seems to be enough to "
            "produce the Phase 3 spill-over.",
            f"- **Over-masking costs B a little outside M through lost evidence, not false alarms:** for dilate4 the outside-M false-alarm "
            f"rate is {far_out['c30__dilate4']:.3f} (oracle {far_out['z30']:.3f}), and outside-M F1 drops from {f1_out['z30']:.3f} to {f1_out['c30__dilate4']:.3f}.",
            f"- **A flags about half of the zero-filled cloud area as change even with the oracle mask** (false-alarm rate in M "
            f"{a_in['z30']:.3f} at 30%, against {a_in['c30']:.3f} with cloud values). This is the Step 4A finding in the new partition.",
        ]
        addendum_obs = "\n".join(obs)
    else:
        anchor, addendum, addendum_obs = "Addendum not yet run.", "_pending_", "_pending_"

    fill = {
        "AUDIT": audit, "PRIMARY_TABLE": "\n".join(rows),
        "B_Z30": f"{f1(B, 'z30'):.3f}", "B_C30": f"{f1(B, 'c30'):.3f}",
        "D_OVER": "/".join(f"{es[f'c30__{v}']['overmasked_clear_frac']:.0%}" for v in DIL), "D_COST": cost(prim, DIL),
        "E_COST": cost(prim, VARIANTS[:3]), "E_MISS": "/".join(f"{es[f'c30__{v}']['missed_cloud_frac']:.0%}" for v in VARIANTS[:3]),
        "M_COST": cost(prim, VARIANTS[6:]), "B_MIN_IMPERFECT": f"{min(f1(B, f'c30__{v}') for v in VARIANTS):.3f}",
        "B_DIL_MAX_OTHER": f"{max(-fam[f'secondary_p2_dropout_{lv}_imperfect_minus_oracle'][v]['mean_diff'] for lv in ('c10', 'c60') for v in DIL):.3f}",
        "B_UNDER_MAX_OTHER": f"{min(r['mean_diff'] for lv in ('c10', 'c60') for r in fam[f'secondary_p2_dropout_{lv}_imperfect_minus_oracle'].values()):+.3f}",
        "A_Z30": f"{f1(A, 'z30'):.3f}", "A_C30": f"{f1(A, 'c30'):.3f}",
        "A_DIL_COST": cost(fam["secondary_p2_full_c30_imperfect_minus_oracle"], DIL),
        "A_OVER_FAR_CLEAN": f"{np.mean(over('p2_full', 'over_clean_false_alarm_rate')):.3f}", "A_OVER_FAR_RANGE": rng(over("p2_full", "over_input_false_alarm_rate")),
        "B_OVER_FAR_CLEAN": f"{np.mean(over('p2_dropout', 'over_clean_false_alarm_rate')):.3f}", "B_OVER_FAR_RANGE": rng(over("p2_dropout", "over_input_false_alarm_rate")),
        "B_OVER_REC": f"{reg['p2_dropout']['c30__dilate4']['over_clean_recall']['mean']:.3f} → {reg['p2_dropout']['c30__dilate4']['over_input_recall']['mean']:.3f}",
        "BA_MIN": f"{min(r['mean_diff'] for r in fam['secondary_B_minus_A_c30'].values()):.3f}",
        "BA_MAX": f"{max(r['mean_diff'] for r in fam['secondary_B_minus_A_c30'].values()):.3f}",
        "MIN_UNDER_OTHER": f"{min(under_other.values()):.3f}",
        "B_MISSED_FAR": rng(b_missed), "B_ERODE2_FAR": rng(e2),
        "ADDENDUM_ANCHOR": anchor, "ADDENDUM_NOTES": addendum, "ADDENDUM_OBS": addendum_obs,
        "TABLES": (OUT / "tables.md").read_text(encoding="utf-8"),
    }
    text = (R / "PHASE4ABIS_REPORT.template.md").read_text(encoding="utf-8")
    for k, v in fill.items():
        text = text.replace("{" + k + "}", v)
    assert "{" not in text.split("## 8.")[0].replace("{lv}", ""), "unfilled placeholder"
    (R / "PHASE4ABIS_REPORT.md").write_text(text, encoding="utf-8")
    print(f"PHASE4ABIS_REPORT.md written; {len(checks)} prose claims checked")


if __name__ == "__main__":
    main()
