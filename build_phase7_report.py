"""Fill experiments/robust/PHASE7_REPORT.template.md -> PHASE7_REPORT.md from phase7/summary.json and p7_audit.json.

Every {KEY} is computed here; the build fails if a qualitative claim in the template is not supported by the results.

Usage:
    .venv/Scripts/python.exe build_phase7_report.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R = ROOT / "experiments" / "robust"
OUT = R / "phase7"
LOGS = ROOT / "logs" / "robust"
NEW_ARMS = ["p7_unet_vh", "p7_unet_vvvh", "p7_dsunet_vh", "p7_dsunet_vvvh"]


def d(r) -> str:
    return f"Δ {r['mean_diff']:+.3f}, 95% CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}], Holm p {r['holm_p']:.2g}"


def main() -> None:
    s = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    au = json.loads((OUT / "p7_audit.json").read_text(encoding="utf-8"))
    hc = subprocess.run([sys.executable, "verify_prereg_hashes.py", "experiments/robust/PHASE7_PREREGISTRATION.md"],
                        cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    A, W, P, SD, SF = s["arms"], s["worldcover"], s["primary_rq3"], s["secondary_dsunet"], s["secondary_far"]
    m = lambda arm, k="f1", cs="9_cities": A[arm][cs][k]
    ms = lambda e: f"{e['mean']:.3f} ± {e['sd']:.3f}"
    cls = lambda k: k.split(", ", 1)[1]
    far_lower = [cls(k) for k, r in SF.items() if r["holm_p"] < 0.05 and r["mean_diff"] < 0]
    far_higher = [cls(k) for k, r in SF.items() if r["holm_p"] < 0.05 and r["mean_diff"] > 0]
    far_none = [cls(k) for k, r in SF.items() if r["holm_p"] >= 0.05]
    tested = [cls(k) for k in SF]
    recall_lower = all(W["p7_unet_vvvh"][c]["recall"]["mean"] < W["p6_unet_sar"][c]["recall"]["mean"] for c in tested)

    # retrained seed 2 against the last step line of the interrupted log (4 decimals is all that log kept)
    last = re.findall(r"step (\d+) loss (\S+) val (\S+)", (LOGS / "p7_run_part1_interrupted.log").read_text(encoding="utf-8"))[-1]
    hist = {h["step"]: h for h in json.loads((R / "p7_unet_vvvh_seed2" / "metrics.json").read_text(encoding="utf-8"))["history"]}
    h = hist[int(last[0])]
    seed2_ok = f"{h['loss']:.4f}" == last[1] and f"{h['val_score']:.4f}" == last[2]
    train_src = (ROOT / "src" / "training" / "train_rrd.py").read_bytes()
    snaps = [R / f"{a}_seed{n}" / "train_rrd.py.snapshot" for a in NEW_ARMS for n in range(1, 11)]
    snaps_ok = all(p.read_bytes() == train_src for p in snaps)
    n_kept = (LOGS / "p7_run.log").read_text(encoding="utf-8").count("exists, skip")
    n_eval = len(list((OUT / "runs").glob("*.json")))

    fill = {
        "HASH_CHECK": (hc.stdout.strip().splitlines() or ["no output"])[-1],
        "AUDIT": f"**{au['n_checks']} checks, {au['n_failed']} failed**",
        "SNAPSHOTS": f"All {len(snaps)} runs' code snapshots are byte-identical to the registered `train_rrd.py`.",
        "N_KEPT": str(n_kept),
        "SEED2_CHECK": f"the retrained run reproduces the interrupted log's last line (step {last[0]}: loss {last[1]}, "
                       f"validation {last[2]}; the log kept 4 decimals)",
        "N_EVAL": str(n_eval), "N_CITIES": str(len(s["cities"])), "CITIES": ", ".join(s["cities"]),
        "L_VVVH": P["VV+VH - VV"]["label"], "D_VVVH": d(P["VV+VH - VV"]),
        "L_VH": P["VH - VV"]["label"], "D_VH": d(P["VH - VV"]),
        "F_U_VV": ms(m("p6_unet_sar")), "F_U_VH": ms(m("p7_unet_vh")), "F_U_VVVH": ms(m("p7_unet_vvvh")),
        "D_DS_VVVH": d(SD["VV+VH - VV"]), "L_DS_VVVH": SD["VV+VH - VV"]["label"],
        "D_DS_VH": d(SD["VH - VV"]), "L_DS_VH": SD["VH - VV"]["label"],
        "F_D_VV": ms(m("p2_full")), "F_D_VH": ms(m("p7_dsunet_vh")), "F_D_VVVH": ms(m("p7_dsunet_vvvh")),
        "N_CLASSES": str(len(SF)), "FAR_LOWER": ", ".join(far_lower), "FAR_NONE": ", ".join(far_none),
        "R_VVVH": ms(m("p7_unet_vvvh", "recall")), "R_VV": ms(m("p6_unet_sar", "recall")),
        "P_VVVH": ms(m("p7_unet_vvvh", "precision")), "P_VV": ms(m("p6_unet_sar", "precision")),
        "F10_U_VVVH": ms(m("p7_unet_vvvh", cs="10_cities")), "F10_U_VV": ms(m("p6_unet_sar", cs="10_cities")),
        "FAR10_U_VVVH": ms(m("p7_unet_vvvh", "false_alarm_rate", "10_cities")),
        "TABLES": (OUT / "tables.md").read_text(encoding="utf-8"),
    }
    claims = {
        "hashes unchanged": hc.returncode == 0 and "0 failed" in fill["HASH_CHECK"],
        "audit 0 failed": au["n_failed"] == 0,
        "60 evaluations saved": n_eval == 60,
        "5 runs kept from the first attempt": n_kept == 5,
        "retrained seed 2 matches the interrupted log": seed2_ok,
        "code snapshots identical": snaps_ok,
        "Las Vegas excluded, 9 cities": "lasvegas" not in s["cities"] and len(s["cities"]) == 9,
        "RQ3: VV+VH lower": P["VV+VH - VV"]["label"] == "VV+VH lower",
        "RQ3: VH lower": P["VH - VV"]["label"] == "VH lower",
        "VH lowers more than VV+VH": P["VH - VV"]["mean_diff"] < P["VV+VH - VV"]["mean_diff"],
        "VV best, VH worst for SAR alone": m("p6_unet_sar")["mean"] > m("p7_unet_vvvh")["mean"] > m("p7_unet_vh")["mean"],
        "DS-UNet: none detectable": all(r["label"] == "no detectable difference" for r in SD.values()),
        "FAR: some lower, none higher": bool(far_lower) and not far_higher,
        "VV+VH recall lower in every class tested": recall_lower,
        "VV+VH recall lower and precision higher overall":
            m("p7_unet_vvvh", "recall")["mean"] < m("p6_unet_sar", "recall")["mean"]
            and m("p7_unet_vvvh", "precision")["mean"] > m("p6_unet_sar", "precision")["mean"],
        "10 cities: VV+VH collapses below VV": m("p7_unet_vvvh", cs="10_cities")["mean"] < m("p6_unet_sar", cs="10_cities")["mean"]
            and m("p7_unet_vvvh", cs="10_cities")["mean"] < m("p7_unet_vvvh")["mean"],
    }
    bad = [k for k, ok in claims.items() if not ok]
    if bad:
        raise SystemExit(f"claims not supported by the results: {bad}")
    text = (R / "PHASE7_REPORT.template.md").read_text(encoding="utf-8")
    for k, v in fill.items():
        text = text.replace("{" + k + "}", v)
    left = re.findall(r"\{[A-Z][A-Za-z0-9_]+\}", text.split("## 6.")[0])
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    (R / "PHASE7_REPORT.md").write_text(text, encoding="utf-8")
    print("PHASE7_REPORT.md written")


if __name__ == "__main__":
    main()
