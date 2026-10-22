"""Fill experiments/robust/PHASE4A_REPORT.template.md -> PHASE4A_REPORT.md and check the numbers quoted in its prose.

Usage:
    .venv/Scripts/python.exe build_phase4a_report.py
"""

import json
from pathlib import Path

R = Path(__file__).resolve().parent / "experiments" / "robust"


def main() -> None:
    s = json.loads((R / "phase4a" / "summary.json").read_text(encoding="utf-8"))
    au = json.loads((R / "phase3" / "p3_audit.json").read_text(encoding="utf-8"))
    a, b = s["arms"]["p2_full"]["clean"], s["arms"]["p2_dropout"]["clean"]
    rec = {arm: [round(100 * s["recovery"][arm][z]["mean"]) for z in ("z10", "z30", "z60")] for arm in s["recovery"]}
    ba = [round(r["mean_diff"], 3) for r in s["paired"]["secondary_B_minus_A_clean"]["rows"].values()]
    reg = s["regions"]
    # every number typed in the prose, against the summary
    quoted = {
        "B recovery 94/90/87": rec["p2_dropout"] == [94, 90, 87],
        "A recovery 40/25/16": rec["p2_full"] == [40, 25, 16],
        "B F1 0.539/0.516/0.499, clean 0.553": [f"{b[z]['f1']['mean']:.3f}" for z in ("z10", "z30", "z60", "clean")] == ["0.539", "0.516", "0.499", "0.553"],
        "B > A +0.146/+0.255/+0.307": ba == [0.146, 0.255, 0.307],
        "A FAR inside zero ~0.51": all(0.50 <= reg["p2_full"][z]["inside_zero_false_alarm_rate"]["mean"] <= 0.53 for z in ("z10", "z30", "z60")),
        "B FAR inside zero ~0.05": all(reg["p2_dropout"][z]["inside_zero_false_alarm_rate"]["mean"] < 0.06 for z in ("z10", "z30", "z60")),
        "B recall inside 0.55-0.61": all(0.545 <= reg["p2_dropout"][z]["inside_zero_recall"]["mean"] <= 0.61 for z in ("z10", "z30", "z60")),
        "primary all Holm < 0.05 and 10/10": all(r["holm_p"] < 0.05 and r["b_gt_a"] == 10 for r in s["paired"]["primary_zero_vs_cloud"]["rows"].values()),
        "B missing (Phase 3) 0.478": f"{json.loads((R / 'phase3' / 'summary.json').read_text(encoding='utf-8'))['arms']['p2_dropout']['clean']['missing']['f1']['mean']:.3f}" == "0.478",
    }
    bad = [k for k, ok in quoted.items() if not ok]
    assert not bad, f"prose numbers disagree with summary.json: {bad}"
    text = (R / "PHASE4A_REPORT.template.md").read_text(encoding="utf-8")
    text = text.replace("<!-- P3_AUDIT -->", f"{au['n_checks']} checks, **{au['n_failed']} failed** (`phase3/p3_audit.json`).")
    text = text.replace("<!-- TABLES -->", (R / "phase4a" / "tables.md").read_text(encoding="utf-8").strip())
    assert "<!--" not in text, "unfilled marker"
    (R / "PHASE4A_REPORT.md").write_text(text, encoding="utf-8")
    print(f"wrote experiments/robust/PHASE4A_REPORT.md ({len(quoted)} quoted numbers checked)")


if __name__ == "__main__":
    main()
