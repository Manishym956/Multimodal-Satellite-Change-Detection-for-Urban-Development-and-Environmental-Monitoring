"""Fill experiments/robust/PHASE8_REPORT.template.md -> PHASE8_REPORT.md from phase8/summary.json and p8_audit.json.

Every {KEY} is computed here; the build fails if a qualitative claim in the template is not supported by the results.

Usage:
    .venv/Scripts/python.exe build_phase8_report.py
"""

import filecmp
import json
import re
import statistics
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R = ROOT / "experiments" / "robust"
OUT = R / "phase8"
GAPS = ["gap01", "gap03", "gap06", "gap12", "gap18"]
MONTHS = {g: int(g[3:]) for g in GAPS}


def d(r) -> str:
    return f"Δ {r['mean_diff']:+.3f}, 95% CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}], Holm p {r['holm_p']:.2g}"


def slope(r) -> str:
    return (f"mean slope {12 * r['mean']:+.4f} F1 per year, 95% CI [{12 * r['ci95'][0]:+.4f}, {12 * r['ci95'][1]:+.4f}], "
            f"t p {r['t_p']:.3f}")


def main() -> None:
    s = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    au = json.loads((OUT / "p8_audit.json").read_text(encoding="utf-8"))
    hc = subprocess.run([sys.executable, "verify_prereg_hashes.py", "experiments/robust/PHASE8_PREREGISTRATION.md"],
                        cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    old = sorted((OUT / "runs_part1_interrupted").iterdir())
    same = [p.name for p in old if filecmp.cmp(p, OUT / "runs" / p.name, shallow=False)]
    A, prim = s["arms"], s["primary_cities"]
    ms = lambda e: f"{e['mean']:.3f} ± {e['sd']:.3f}"
    fB = {t: A["p2_dropout"][t]["f1"] for t in ["oscd"] + GAPS}
    P, SB, SA, SS, SR = s["primary_rq2"], s["secondary_B_vs_sar"], s["secondary_async_vs_sar"], s["secondary_source"], s["rq5_landcover"]
    AS, SEN = s["secondary_async_slope"], s["sensitivity_slope_all_cities"]
    days = [abs((date.fromisoformat(v["day"]) - date.fromisoformat(v["target"])).days)
            for v in s["scenes"].values() if v["city"] in prim and v["tag"] in GAPS]
    before = {g: [c for c in s["image_before_oscd_t1"][g] if c in prim] for g in GAPS}
    async_hit = [k for k, r in SA.items() if r["holm_p"] < 0.05]
    rq5_hit = [k for k, r in SR.items() if r["holm_p"] < 0.05]
    cls = lambda k: k.split(", ", 1)[1]
    share = {cls(k): s["worldcover_share"][cls(k)] for k in SR}
    B_d = [r["mean_diff"] for r in SB.values()]
    fill = {
        "HASH_CHECK": (hc.stdout.strip().splitlines() or ["no output"])[-1],
        "AUDIT": f"**{au['n_checks']} checks, {au['n_failed']} failed**",
        "DETERMINISM": f"the re-run reproduced all {len(old)} interrupted output files byte for byte ({len(same)}/{len(old)} identical)",
        "N_PRIMARY": str(len(prim)), "PRIMARY": ", ".join(prim), "N_ALL": str(len(s["available_cities"])),
        "DAYS_MED": f"{statistics.median(days):g}", "DAYS_MAX": str(max(days)), "MAX_CLOUD": f"{s['max_cloud']:.2f}",
        "BEFORE_GAP12": f"{len(before['gap12'])} of {len(prim)}" + (f" ({', '.join(before['gap12'])})" if before["gap12"] else ""),
        "BEFORE_GAP18": f"{len(before['gap18'])} of {len(prim)} ({', '.join(before['gap18'])})",
        "RQ2_LINE": slope(P), "RQ2_LABEL": P["label"], "RQ2_WILCOXON": f"{P['wilcoxon_p']:.3f}",
        "RQ2_POS": str(sum(x > 0 for x in P["slopes_f1_per_month"])),
        "F_B_OSCD": ms(fB["oscd"]), **{f"F_B_{g[3:]}": ms(fB[g]) for g in GAPS},
        "SENS_LINE": slope(SEN), "SENS_LABEL": SEN["label"],
        "B_SAR_SUMMARY": f"no detectable difference at any age (mean Δ from {min(B_d):+.3f} to {max(B_d):+.3f}, smallest Holm p "
                         f"{min(r['holm_p'] for r in SB.values()):.2f}).",
        "F_SAR": ms(A["p6_unet_sar"]["oscd"]["f1"]),
        "ASYNC_SAR_SUMMARY": f"async lower at 1 month ({d(SA['async@gap01 - SAR-only'])}); no detectable difference at 3, 6, "
                             "12 and 18 months.",
        "ASYNC_SLOPE_LINE": slope(AS), "ASYNC_SLOPE_LABEL": AS["label"],
        "D_SOURCE": d(SS["B@gee_t1 - B@oscd"]), "SOURCE_LABEL": SS["B@gee_t1 - B@oscd"]["label"],
        "D_NATIVE": d(SS["B@oscd - SAR-only"]), "NATIVE_LABEL": SS["B@oscd - SAR-only"]["label"],
        "RQ5_SUMMARY": f"B lower in shrubland ({d(SR['B@oscd - SAR-only, shrubland'])}); no detectable difference in the "
                       f"other {len(SR) - 1} classes (built-up: {d(SR['B@oscd - SAR-only, built-up'])}).",
        "WC_SHARE": "; ".join(f"{c} {100 * v:.1f}%" for c, v in share.items()),
        "SHRUB_PCT": f"{100 * share['shrubland']:.1f}",
        "TABLES": (OUT / "tables.md").read_text(encoding="utf-8"),
    }
    claims = {
        "hashes unchanged": hc.returncode == 0 and "0 failed" in fill["HASH_CHECK"],
        "audit 0 failed": au["n_failed"] == 0,
        "16 interrupted files reproduced": len(old) == 16 and len(same) == 16,
        "chongqing excluded from primary only": "chongqing" not in prim and "chongqing" in s["available_cities"],
        "no primary image before date 1 at 1-6 months": not any(before[g] for g in ("gap01", "gap03", "gap06")),
        "RQ2: no detectable effect": P["label"] == "no detectable effect of the gap" and P["t_p"] >= 0.05,
        "RQ2: t and Wilcoxon disagree": P["wilcoxon_p"] < 0.05,
        "RQ2: mean slope positive": P["mean"] > 0,
        "youngest image scores lowest for B": min(fB, key=lambda t: fB[t]["mean"]) == "gap01",
        "sensitivity: no detectable effect": SEN["label"] == "no detectable effect of the gap",
        "B vs SAR: none detectable": all(r["label"] == "no detectable difference" for r in SB.values()),
        "async vs SAR: only gap01, lower": async_hit == ["async@gap01 - SAR-only"] and SA[async_hit[0]]["label"] == "async lower",
        "async slope: no detectable effect": AS["label"] == "no detectable effect of the gap",
        "source family: neither detectable": all(r["label"] == "no detectable difference" for r in SS.values()),
        "RQ5: only shrubland, lower": rq5_hit == ["B@oscd - SAR-only, shrubland"] and SR[rq5_hit[0]]["mean_diff"] < 0,
        "built-up is the largest class tested": max(share, key=share.get) == "built-up",
        "shrubland under 2% of pixels": share["shrubland"] < 0.02,
    }
    bad = [k for k, ok in claims.items() if not ok]
    if bad:
        raise SystemExit(f"claims not supported by summary.json: {bad}")
    text = (R / "PHASE8_REPORT.template.md").read_text(encoding="utf-8")
    for k, v in fill.items():
        text = text.replace("{" + k + "}", v)
    left = re.findall(r"\{[A-Z][A-Za-z0-9_]+\}", text.split("## 7.")[0])
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    (R / "PHASE8_REPORT.md").write_text(text, encoding="utf-8")
    print("PHASE8_REPORT.md written")


if __name__ == "__main__":
    main()
