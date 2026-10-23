"""Fill experiments/robust/PHASE6_REPORT.template.md -> PHASE6_REPORT.md from phase6/summary.json and p6_audit.json.

Every {KEY} is computed here; the build fails if a qualitative claim in the template is not supported by the results.

Usage:
    .venv/Scripts/python.exe build_phase6_report.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R = ROOT / "experiments" / "robust"
OUT = R / "phase6"


def d(r) -> str:
    return f"Δ {r['mean_diff']:+.3f}, 95% CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}], Holm p {r['holm_p']:.2g}"


def table(rows) -> str:
    L = ["| contrast | mean Δ | 95% CI | d_z | Δ > 0 | Holm p (t) | Holm p (Wilcoxon) | label |", "|---|---|---|---|---|---|---|---|"]
    for k, r in rows.items():
        L.append(f"| {k} | {r['mean_diff']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['dz']:+.2f} | {r['b_gt_a']}/10 | "
                 f"{r['holm_p']:.1e} | {r['holm_wilcoxon_p']:.4f} | {r['label']} |")
    return "\n".join(L)


def main() -> None:
    s = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    au = json.loads((OUT / "p6_audit.json").read_text(encoding="utf-8"))
    hc = subprocess.run([sys.executable, "verify_prereg_hashes.py", "experiments/robust/PHASE6_PREREGISTRATION.md"],
                        cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    A = s["arms"]
    f = lambda arm, c="full", thr="at_selected": A[arm][thr][c]["f1"]
    ms = lambda e: f"{e['mean']:.3f} ± {e['sd']:.3f}"
    P4, P1, SF, SA = s["primary_rq4"], s["primary_rq1"], s["secondary_fusion"], s["secondary_async"]
    thr = [t for a in ("p6_unet_sar", "p6_unet_opt", "p6_unet_fusion") for t in s["selected_threshold"][a]["full"]]
    fill = {
        "HASH_CHECK": (hc.stdout.strip().splitlines() or ["no output"])[-1],
        "AUDIT": f"**{au['n_checks']} checks, {au['n_failed']} failed**",
        "RQ4_TABLE": table(P4), "RQ1_TABLE": table(P1),
        "D_SAR": d(P4["A - SAR-only"]), "D_OPT": d(P4["A - optical-only"]), "D_FUS": d(P4["A - input fusion"]),
        "F_A": ms(f("p2_full")), "F_SAR": ms(f("p6_unet_sar")), "F_OPT": ms(f("p6_unet_opt")), "F_FUS": ms(f("p6_unet_fusion")),
        "D_FUS_OPT": d(SF["fusion - optical-only"]), "D_FUS_SAR": d(SF["fusion - SAR-only"]), "D_OPT_SAR": d(SF["optical-only - SAR-only"]),
        "D_B_SAR": d(P1["B@no_o2 - SAR-only"]), "D_ASYNC_SAR": d(P1["async@no_o2 - SAR-only"]),
        "D_ASYNC_B": d(SA["async@no_o2 - B@no_o2"]), "D_B_NOOPT": d(SA["B@no_o2 - B@no_opt"]),
        "ASYNC_NOOPT": ms(f("p6_async", "no_opt")), "ASYNC_NOO2": ms(f("p6_async", "no_o2")),
        "F05_LINE": "; ".join(f"{a} {f(a, thr='at_0.5')['mean']:.3f}" for a in A),
        "THR_MIN": f"{min(thr):.2f}", "THR_MAX": f"{max(thr):.2f}",
        "TABLES": (OUT / "tables.md").read_text(encoding="utf-8"),
    }
    claims = {
        "hashes unchanged": hc.returncode == 0 and "0 failed" in fill["HASH_CHECK"],
        "audit 0 failed": au["n_failed"] == 0,
        "RQ4: DS-UNet better against all three": all(r["label"] == "DS-UNet better" for r in P4.values()),
        "fusion vs optical not detectable": SF["fusion - optical-only"]["holm_p"] >= 0.05,
        "fusion and optical beat SAR": all(SF[k]["holm_p"] < 0.05 and SF[k]["mean_diff"] > 0 for k in ("fusion - SAR-only", "optical-only - SAR-only")),
        "RQ1: no detectable difference for both": all(r["label"] == "no detectable difference" for r in P1.values()),
        "secondary async family: neither detectable": all(r["holm_p"] >= 0.05 for r in SA.values()),
        "async scores higher without the historical image": f("p6_async", "no_opt")["mean"] > f("p6_async", "no_o2")["mean"],
        "SAR-only U-Net above the companion's 0.241 and 0.309": f("p6_unet_sar")["mean"] > 0.309,
    }
    bad = [k for k, ok in claims.items() if not ok]
    if bad:
        raise SystemExit(f"claims not supported by summary.json: {bad}")
    text = (R / "PHASE6_REPORT.template.md").read_text(encoding="utf-8")
    for k, v in fill.items():
        text = text.replace("{" + k + "}", v)
    left = re.findall(r"\{[A-Z][A-Za-z0-9_]+\}", text.split("## 6.")[0])
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    (R / "PHASE6_REPORT.md").write_text(text, encoding="utf-8")
    print("PHASE6_REPORT.md written")


if __name__ == "__main__":
    main()
