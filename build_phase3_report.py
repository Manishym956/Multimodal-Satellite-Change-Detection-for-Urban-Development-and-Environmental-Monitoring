"""Fill experiments/robust/PHASE3_REPORT.template.md from phase3/summary.json and phase3/tables.md -> PHASE3_REPORT.md.

Usage:
    .venv/Scripts/python.exe build_phase3_report.py
"""

import json
from pathlib import Path

R = Path(__file__).resolve().parent / "experiments" / "robust"


def fmt_p(p: float) -> str:
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def main() -> None:
    s = json.loads((R / "phase3" / "summary.json").read_text(encoding="utf-8"))
    rows = []
    for c, r in s["paired"]["primary_clean"]["rows"].items():
        a, b = s["arms"]["p2_full"]["clean"][c]["f1"], s["arms"]["p2_dropout"]["clean"][c]["f1"]
        rows.append(f"| {c} | {a['mean']:.3f} ± {a['sd']:.3f} | {b['mean']:.3f} ± {b['sd']:.3f} | "
                    f"{r['mean_diff']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {fmt_p(r['holm_p'])} | {r['b_gt_a']}/10 |")
    text = (R / "PHASE3_REPORT.template.md").read_text(encoding="utf-8")
    text = text.replace("<!-- PRIMARY_ROWS -->", "\n".join(rows))
    text = text.replace("<!-- TABLES -->", (R / "phase3" / "tables.md").read_text(encoding="utf-8").strip())
    assert "<!--" not in text, "unfilled marker"
    (R / "PHASE3_REPORT.md").write_text(text, encoding="utf-8")
    print("wrote experiments/robust/PHASE3_REPORT.md")


if __name__ == "__main__":
    main()
