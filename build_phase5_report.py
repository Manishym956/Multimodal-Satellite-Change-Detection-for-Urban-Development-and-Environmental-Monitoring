"""Fill experiments/robust/PHASE5_REPORT.template.md -> PHASE5_REPORT.md from phase5/summary.json and p5_audit.json.

Generated parts: integrity facts, the primary table (with pre-specified labels; row format checked by audit_p5.py),
secondary and descriptive summaries, and the analysis tables pasted unchanged. Interpretation text lives in the template and
must quote only numbers that are checked here: every {KEY} placeholder is computed from summary.json, and the build fails if
a qualitative claim is unsupported.

Usage:
    .venv/Scripts/python.exe build_phase5_report.py
    .venv/Scripts/python.exe audit_p5.py      # then re-run: checks this report's primary table
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R = ROOT / "experiments" / "robust"
OUT = R / "phase5"
LEVELS = ("c10", "c30", "c60")
DETS = ("rb95", "rb90", "rb99", "bal", "default")


def rng(xs) -> str:
    return f"{min(xs):+.3f} to {max(xs):+.3f}"


def main() -> None:
    s = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    P, A, C, CB = s["primary"], s["secondary_A"], s["secondary_C"], s["secondary_C_minus_B"]
    OM, FD = s["secondary_B_other_masks"], s["secondary_B_false_detections"]
    f1 = lambda arm, c: s["arms"][arm][c]["f1"]["mean"]
    B = lambda c: f1("p2_dropout", c)
    rg = s["retained_gain"]
    dt = s["detector_test"]
    io = s["inside_outside"]["p2_dropout"]

    rows = ["| contrast | mean Δ | 95% CI | d_z | Δ > 0 | Holm p (t) | Holm p (Wilcoxon) | label |", "|---|---|---|---|---|---|---|---|"]
    for k, r in P.items():
        rows.append(f"| {k} | {r['mean_diff']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['dz']:+.2f} | "
                    f"{r['b_gt_a']}/10 | {r['holm_p']:.1e} | {r['holm_wilcoxon_p']:.4f} | {r['label']} |")

    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((OUT / "runs").glob("*.json"))]
    hc = subprocess.run([sys.executable, "verify_prereg_hashes.py", "experiments/robust/PHASE5_PREREGISTRATION.md"],
                        cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    hash_line = (hc.stdout.strip().splitlines() or ["no output"])[-1]
    au_path = OUT / "p5_audit.json"
    au = json.loads(au_path.read_text(encoding="utf-8")) if au_path.exists() else None
    if au is None:
        audit = "not yet run."
    else:
        only_report = au["failed"] and all("REPORT.md exists" in f["check"] for f in au["failed"])
        audit = f"**{au['n_checks']} checks, {au['n_failed']} failed**" + (
            " (only the report-table check, because this report did not exist yet; re-run after building)." if only_report else ".")
    man = json.loads((OUT / "composite_manifest.json").read_text(encoding="utf-8"))["entries"]
    n_bank_test = sum(e["in_bank"] for e in man if e["split"] == "test")
    nb = s["C_minus_B_non_bank_composites"]

    fill = {
        "PRIMARY_TABLE": "\n".join(rows), "AUDIT": audit, "HASH_CHECK": hash_line,
        "N_ANCHOR_RUNS": str(sum(all(r["anchors"].values()) for r in runs)),
        "B_F1_LINE": "; ".join(f"{lv}: cloud {B(f'{lv}__cloud'):.3f}, bal {B(f'{lv}__bal'):.3f}, rb95 {B(f'{lv}__rb95'):.3f}, "
                               f"oracle {B(f'{lv}__oracle'):.3f}" for lv in LEVELS) + f"; clean {B('clean'):.3f}",
        "A_D_bal": ", ".join(f"{A[f'{lv}: rb95 - bal']['mean_diff']:+.3f}" for lv in LEVELS),
        "C_D_bal": ", ".join(f"{lv} {C[f'{lv}: rb95 - bal']['mean_diff']:+.3f} (Holm p {C[f'{lv}: rb95 - bal']['holm_p']:.2g})" for lv in LEVELS),
        "CB_RANGE": rng([r["mean_diff"] for r in CB.values()]),
        "N_BANK_TEST": str(n_bank_test),
        "CB_NONBANK_rb95": ", ".join(f"{lv} {nb[lv]['rb95']['mean']:+.3f} ± {nb[lv]['rb95']['sd']:.3f} ({nb[lv]['n_composites']})" for lv in LEVELS),
        "OTHER_MASKS": "; ".join(f"{k} {r['mean_diff']:+.3f} (Holm p {r['holm_p']:.2g})" for k, r in OM.items()),
        "CLEAN_COST": ", ".join(f"{k.split(' - ')[0].split('__')[1]} {r['mean_diff']:+.3f}" for k, r in FD.items()),
        "CLEAN_rb95": f"{-FD['clean__rb95 - clean']['mean_diff']:.3f}",
        "DT_rb95_recall": ", ".join(f"{dt[lv]['rb95']['recall']:.3f}" for lv in LEVELS),
        "DT_rb95_over": ", ".join(f"{dt[lv]['rb95']['overmasked_clear']:.3f}" for lv in LEVELS),
        "DT_bal_recall": ", ".join(f"{dt[lv]['bal']['recall']:.3f}" for lv in LEVELS),
        "DT_bal_over": ", ".join(f"{dt[lv]['bal']['overmasked_clear']:.3f}" for lv in LEVELS),
        "DT_rb99_over": ", ".join(f"{dt[lv]['rb99']['overmasked_clear']:.3f}" for lv in LEVELS),
        "DT_MISS_RANGE": f"{100 * (1 - dt['c10']['rb95']['recall']):.0f}–{100 * (1 - dt['c30']['rb95']['recall']):.0f}%",
        "B_rb99": ", ".join(f"{B(f'{lv}__rb99'):.3f}" for lv in LEVELS),
        "B_c30_rb95": f"{B('c30__rb95'):.3f}", "B_c30_bal": f"{B('c30__bal'):.3f}",
        "IN_rb95_vs_bal": ", ".join(f"{io[f'{lv}__rb95']['inside_f1']['mean']:.3f} vs {io[f'{lv}__bal']['inside_f1']['mean']:.3f}" for lv in LEVELS),
        "OUT_rb95_vs_bal": ", ".join(f"{io[f'{lv}__rb95']['outside_f1']['mean']:.3f} vs {io[f'{lv}__bal']['outside_f1']['mean']:.3f}" for lv in LEVELS),
        "OUT_oracle_c30": f"{io['c30__oracle']['outside_f1']['mean']:.3f}", "OUT_rb95_c30": f"{io['c30__rb95']['outside_f1']['mean']:.3f}",
        "RG_B_LINE": "; ".join(f"{lv}: " + ", ".join(f"{d} {rg['p2_dropout'][lv][d]['mean']:.2f}" for d in DETS) for lv in LEVELS),
        "ORACLE_GAP_RANGE": f"{min(-P[f'{lv}: rb95 - oracle']['mean_diff'] for lv in LEVELS):.3f}–{max(-P[f'{lv}: rb95 - oracle']['mean_diff'] for lv in LEVELS):.3f}",
        "TABLES": (OUT / "tables.md").read_text(encoding="utf-8"),
    }
    for lv in LEVELS:
        fill[f"RG_B_{lv}_rb95"] = f"{rg['p2_dropout'][lv]['rb95']['mean']:.3f}"
        for ref in ("bal", "oracle", "cloud"):
            fill[f"D_{lv}_{ref}"] = f"{P[f'{lv}: rb95 - {ref}']['mean_diff']:+.3f}"

    lab = lambda lv, ref: P[f"{lv}: rb95 - {ref}"]["label"]
    sig = lambda r: r["holm_p"] < 0.05
    rg_c30 = rg["p2_dropout"]["c30"]["rb95"]["mean"]
    # Claims the template's prose makes; the build fails if the results do not support them.
    claims = {
        "pre-registered hashes still match": hc.returncode == 0 and "0 failed" in hash_line,
        "30 runs, all anchors pass, one source-hash set": len(runs) == 30 and fill["N_ANCHOR_RUNS"] == "30"
            and len({json.dumps(r["source_sha256"], sort_keys=True) for r in runs}) == 1,
        "audit has no failure other than the report check": au is not None and all("REPORT.md exists" in f["check"] for f in au["failed"]),
        "primary: 9 contrasts": len(P) == 9,
        "rb95 better than bal at c30 and c60, n.s. at c10": lab("c30", "bal") == lab("c60", "bal") == "rb95 better" and lab("c10", "bal") == "no detectable difference",
        "rb95 worse than oracle at every level": all(lab(lv, "oracle") == "rb95 worse" for lv in LEVELS),
        "rb95 better than cloud at every level": all(lab(lv, "cloud") == "rb95 better" for lv in LEVELS),
        "decision: c30 retained gain in [0.5, 0.75), shown as 0.740": 0.5 <= rg_c30 < 0.75 and fill["RG_B_c30_rb95"] == "0.740",
        "A: rb95 - bal negative and Holm-significant at every level": all(A[f"{lv}: rb95 - bal"]["mean_diff"] < 0 and sig(A[f"{lv}: rb95 - bal"]) for lv in LEVELS),
        "A: rb95 - cloud negative and significant at c10, c30": all(A[f"{lv}: rb95 - cloud"]["mean_diff"] < 0 and sig(A[f"{lv}: rb95 - cloud"]) for lv in ("c10", "c30")),
        "A retained gain unstable (SD > 1 somewhere)": any(rg["p2_full"][lv][d]["sd"] > 1 for lv in LEVELS for d in DETS),
        "C: same labels pattern as B for rb95 - bal": all(sig(C[f"{lv}: rb95 - bal"]) and C[f"{lv}: rb95 - bal"]["mean_diff"] > 0 for lv in ("c30", "c60"))
            and not sig(C["c10: rb95 - bal"]),
        "C - B: no Holm p < 0.05": not any(sig(r) for r in CB.values()),
        "non-bank composites total 34": sum(nb[lv]["n_composites"] for lv in LEVELS) == 34 == 90 - n_bank_test,
        "every clean-image detector mask lowers F1, Holm p < 0.05": all(r["mean_diff"] < 0 and sig(r) for r in FD.values()),
        "test rb95 recall < 0.95 at c10, c30": all(dt[lv]["rb95"]["recall"] < 0.95 for lv in ("c10", "c30")),
        "rb95 masks more than half of clear test pixels at every level": all(dt[lv]["rb95"]["overmasked_clear"] > 0.5 for lv in LEVELS),
        "rb99 masks > 0.99 of clear test pixels": all(dt[lv]["rb99"]["overmasked_clear"] > 0.99 for lv in LEVELS),
        "B rb99 F1 nearly constant over levels (range < 0.002)": max(B(f"{lv}__rb99") for lv in LEVELS) - min(B(f"{lv}__rb99") for lv in LEVELS) < 0.002,
        "B c30: rb99 > rb95 > bal": B("c30__rb99") > B("c30__rb95") > B("c30__bal"),
        "B c60: rb99 >= rb95": B("c60__rb99") >= B("c60__rb95"),
        "rb95 vs bal: higher F1 inside M, lower outside, every level":
            all(io[f"{lv}__rb95"]["inside_f1"]["mean"] > io[f"{lv}__bal"]["inside_f1"]["mean"]
                and io[f"{lv}__rb95"]["outside_f1"]["mean"] < io[f"{lv}__bal"]["outside_f1"]["mean"] for lv in LEVELS),
        "A: rb95 - cloud not significant at c60": not sig(A["c60: rb95 - cloud"]),
    }
    bad = [k for k, ok in claims.items() if not ok]
    if bad:
        raise SystemExit(f"prose claims not supported by summary.json: {bad}")
    text = (R / "PHASE5_REPORT.template.md").read_text(encoding="utf-8")
    for k, v in fill.items():
        text = text.replace("{" + k + "}", v)
    left = re.findall(r"\{[A-Z][A-Za-z0-9_]+\}", text.split("## 9.")[0])
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    (R / "PHASE5_REPORT.md").write_text(text, encoding="utf-8")
    print("PHASE5_REPORT.md written")


if __name__ == "__main__":
    main()
