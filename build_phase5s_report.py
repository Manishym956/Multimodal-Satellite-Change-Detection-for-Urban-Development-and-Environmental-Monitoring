"""Fill experiments/robust/PHASE5S_REPORT.template.md -> PHASE5S_REPORT.md from phase5s/summary.json and p5s_audit.json.

Generated parts: integrity facts, the primary table (with pre-specified labels), secondary summaries, retained gain, curve
facts, and the analysis tables pasted unchanged. Interpretation text lives in the template and must quote only numbers
that are checked here: every {KEY} placeholder is computed from summary.json, and the build fails if a claim is unsupported.

Usage:
    .venv/Scripts/python.exe build_phase5s_report.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R = ROOT / "experiments" / "robust"
OUT = R / "phase5s"
VARIANTS = ("erode2", "erode4", "erode8", "miss10", "miss25", "miss50")
ERODE, MISS = VARIANTS[:3], VARIANTS[3:]
LEVELS = ("c10", "c30", "c60")


def rng(xs) -> str:
    return f"{min(xs):+.3f} to {max(xs):+.3f}"


def main() -> None:
    s = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    rs = s["radius_selection"]
    r_star = rs["r_star"]
    grid = sorted(int(r) for r in rs["val_score_by_radius"])
    val = [rs["val_score_by_radius"][str(r)] for r in grid]
    B = s["arms"]["p2_dropout"]
    P, BC, A, C, O = (s[k] for k in ("primary", "secondary_B_c10_c60", "secondary_A", "secondary_C", "secondary_B_vs_oracle"))
    rg = s["retained_gain"]["p2_dropout"]

    rows = ["| condition | B F1, P (r = 0) | B F1, P dilated (r*) | mean Δ | 95% CI | d_z | Δ > 0 | Holm p (t) | Holm p (Wilcoxon) | label |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        r, a, b = P[f"c30__{v}"], B[f"c30__{v}__0"], B[f"c30__{v}__{r_star}"]
        rows.append(f"| c30__{v} | {a['mean']:.3f} ± {a['sd']:.3f} | {b['mean']:.3f} ± {b['sd']:.3f} | {r['mean_diff']:+.3f} | "
                    f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['dz']:+.2f} | {r['b_gt_a']}/10 | {r['holm_p']:.1e} | "
                    f"{r['holm_wilcoxon_p']:.4f} | {r['label']} |")

    au = json.loads((OUT / "p5s_audit.json").read_text(encoding="utf-8"))
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((OUT / "runs").glob("*.json"))]
    hc = subprocess.run([sys.executable, "verify_prereg_hashes.py", "experiments/robust/PHASE5S_PREREGISTRATION.md"],
                        cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    hash_line = (hc.stdout.strip().splitlines() or ["no output"])[-1]

    curve = {v: [B[f"c30__{v}__{r}"]["mean"] for r in grid] for v in ERODE}
    peak_r = {v: grid[max(range(len(grid)), key=lambda i: curve[v][i])] for v in ERODE}
    over = {v: max(curve[v]) - B[f"c30__{v}__{r_star}"]["mean"] for v in ERODE}

    fill = {
        "PRIMARY_TABLE": "\n".join(rows),
        "AUDIT": f"**{au['n_checks']} checks, {au['n_failed']} failed**",
        "HASH_CHECK": hash_line,
        "N_ANCHOR_RUNS": str(sum(all(r["anchors"].values()) for r in runs)),
        "RSTAR": str(r_star),
        "VAL_CURVE": ", ".join(f"r = {r}: {x:.3f}" for r, x in zip(grid, val)),
        "N_BC_BETTER": str(sum(r["holm_p"] < 0.05 and r["mean_diff"] > 0 for r in BC.values())),
        "BC_ERODE_RANGE": rng([BC[f"{L}__{v}"]["mean_diff"] for L in ("c10", "c60") for v in ERODE]),
        "BC_MISS_RANGE": rng([BC[f"{L}__{v}"]["mean_diff"] for L in ("c10", "c60") for v in MISS]),
        "N_A_MISS_NEG": str(sum(A[f"{L}__{v}"]["mean_diff"] < 0 for L in LEVELS for v in MISS)),
        "N_A_MISS_SIG": str(sum(A[f"{L}__{v}"]["mean_diff"] < 0 and A[f"{L}__{v}"]["holm_p"] < 0.05 for L in LEVELS for v in MISS)),
        "A_ERODE_RANGE": rng([A[f"{L}__{v}"]["mean_diff"] for L in LEVELS for v in ERODE]),
        "N_C_ERODE_SIG": str(sum(C[f"{L}__{v}"]["mean_diff"] > 0 and C[f"{L}__{v}"]["holm_p"] < 0.05 for L in LEVELS for v in ERODE)),
        "C_ERODE_RANGE": rng([C[f"{L}__{v}"]["mean_diff"] for L in LEVELS for v in ERODE]),
        "C_MISS_C30": "; ".join(f"{v} {C[f'c30__{v}']['mean_diff']:+.3f} (Holm p {C[f'c30__{v}']['holm_p']:.2g})" for v in MISS),
        "MAX_OVERDILATION": f"{max(over.values()):.3f}",
        "TABLES": (OUT / "tables.md").read_text(encoding="utf-8"),
    }
    for v in VARIANTS:
        fill[f"D_{v}"] = f"{P[f'c30__{v}']['mean_diff']:+.3f}"
        fill[f"H_{v}"] = f"{P[f'c30__{v}']['holm_p']:.3f}"
        fill[f"HW_{v}"] = f"{P[f'c30__{v}']['holm_wilcoxon_p']:.3f}"
        fill[f"O_{v}"] = f"{O[f'c30__{v}']['mean_diff']:+.3f}"
        fill[f"RG0_{v}"] = f"{rg[f'c30__{v}']['r0']['mean']:.2f}"
        fill[f"RGS_{v}"] = f"{rg[f'c30__{v}']['r_star']['mean']:.2f}"
    for v in ERODE:
        fill[f"PEAK_{v}"], fill[f"PEAKR_{v}"] = f"{max(curve[v]):.3f}", str(peak_r[v])
        fill[f"AT_RSTAR_{v}"] = f"{B[f'c30__{v}__{r_star}']['mean']:.3f}"

    # Claims the template's prose makes; the build fails if the results do not support them.
    claims = {
        "pre-registered hashes still match": hc.returncode == 0 and "0 failed" in hash_line,
        "audit 0 failed": au["n_failed"] == 0,
        "all 30 runs passed every anchor": fill["N_ANCHOR_RUNS"] == "30" and len(runs) == 30,
        "r* is the largest radius in the grid": r_star == grid[-1],
        "validation score still rising at r*": val[-1] > val[-2],
        "erode primaries 'dilation better', 10/10 seeds": all(P[f"c30__{v}"]["label"] == "dilation better" and P[f"c30__{v}"]["b_gt_a"] == 10 for v in ERODE),
        "miss10, miss25 'no detectable difference'": all(P[f"c30__{v}"]["label"] == "no detectable difference" for v in ("miss10", "miss25")),
        "miss50 'dilation better' but |Δ| < 0.005 and Holm Wilcoxon >= 0.05":
            P["c30__miss50"]["label"] == "dilation better" and abs(P["c30__miss50"]["mean_diff"]) < 0.005 and P["c30__miss50"]["holm_wilcoxon_p"] >= 0.05,
        "c10__miss10 is the only B c10/c60 contrast without Holm p < 0.05":
            [k for k, r in BC.items() if r["holm_p"] >= 0.05] == ["c10__miss10"],
        "other B c10/c60 missed-cloud contrasts: Δ > 0 in 10/10 seeds":
            all(BC[f"{L}__{v}"]["b_gt_a"] == 10 for L in ("c10", "c60") for v in MISS if f"{L}__{v}" != "c10__miss10"),
        "no B c10/c60 contrast significantly negative": not any(r["holm_p"] < 0.05 and r["mean_diff"] < 0 for r in BC.values()),
        "all 18 vs-oracle contrasts negative": all(r["mean_diff"] < 0 for r in O.values()) and len(O) == 18,
        "c30 eroded residual vs oracle > -0.05": all(O[f"c30__{v}"]["mean_diff"] > -0.05 for v in ERODE),
        "c30 missed residual vs oracle < -0.09": all(O[f"c30__{v}"]["mean_diff"] < -0.09 for v in MISS),
        "A: all 9 missed-cloud contrasts negative": fill["N_A_MISS_NEG"] == "9",
        "A: no eroded contrast survives Holm": all(A[f"{L}__{v}"]["holm_p"] >= 0.05 for L in LEVELS for v in ERODE),
        "A retained gain unstable (seed SD > 0.3 somewhere)": any(x["r0"]["sd"] > 0.3 for x in s["retained_gain"]["p2_full"].values()),
        "C: all 9 eroded contrasts better after Holm": fill["N_C_ERODE_SIG"] == "9",
        "C at c30 missed: no positive Holm-significant contrast": not any(C[f"c30__{v}"]["mean_diff"] > 0 and C[f"c30__{v}"]["holm_p"] < 0.05 for v in MISS),
        "B c30 eroded retained gain at r* >= 0.85": all(rg[f"c30__{v}"]["r_star"]["mean"] >= 0.85 for v in ERODE),
        "B c30 missed retained gain changes by < 0.01": all(abs(rg[f"c30__{v}"]["r_star"]["mean"] - rg[f"c30__{v}"]["r0"]["mean"]) < 0.01 for v in MISS),
        "erode2 test curve peaks below r*": peak_r["erode2"] < r_star,
        "erode8 test curve still rising at r*": curve["erode8"][-1] > curve["erode8"][-2],
    }
    bad = [k for k, ok in claims.items() if not ok]
    if bad:
        raise SystemExit(f"prose claims not supported by summary.json: {bad}")
    text = (R / "PHASE5S_REPORT.template.md").read_text(encoding="utf-8")
    for k, v in fill.items():
        text = text.replace("{" + k + "}", v)
    left = re.findall(r"\{[A-Z][A-Za-z0-9_]+\}", text.split("## 8.")[0])
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    (R / "PHASE5S_REPORT.md").write_text(text, encoding="utf-8")
    print("PHASE5S_REPORT.md written")


if __name__ == "__main__":
    main()
