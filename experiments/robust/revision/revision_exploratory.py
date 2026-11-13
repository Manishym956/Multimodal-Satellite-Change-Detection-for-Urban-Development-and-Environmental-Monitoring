"""Exploratory contrasts quoted in the revised manuscript (chosen after the results were seen; not pre-registered).

Reads only stored outputs: Step 6/9 summaries (seed lists) and Step 11 real-cloud summary. Paired over seeds, paired
t-test, no multiplicity correction (descriptive). Writes revision_exploratory.json next to this file.

Usage (repository root):  .venv/Scripts/python.exe experiments/robust/revision/revision_exploratory.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from analyze_phase3 import paired  # noqa: E402

R = ROOT / "experiments" / "robust"
load = lambda p: json.loads((R / p).read_text(encoding="utf-8"))


def main() -> None:
    s9, s11 = load("step9/summary.json"), load("realcloud/summary.json")
    v9 = lambda arm, c, k="f1", thr="at_selected": s9["arms"][arm][thr][c][k]["values"]
    E, B, S = "p9_fusion_dropout", "p2_dropout", "p6_unet_sar"
    out = {"step9": {
        **{f"E {k} present - absent": paired(v9(E, "no_opt", k), v9(E, "no_o2", k)) for k in ("precision", "recall", "false_alarm_rate")},
        "E@no_opt - SAR-only": paired(v9(S, "full"), v9(E, "no_opt")),
        "E@no_opt - B@no_opt": paired(v9(B, "no_opt"), v9(E, "no_opt")),
        "B@no_opt - SAR-only": paired(v9(S, "full"), v9(B, "no_opt"))}}
    v11 = lambda set_, arm, c: s11["arms"][set_][arm][c]["f1"]["values"]
    out["step11"] = {f"{st}: B missing - SAR-only": paired(v11(st, S, "oscd"), v11(st, B, "missing")) for st in ("cloudy", "overcast", "all")}
    for v in out.values():
        for r in v.values():
            r.pop("diffs", None)
    (HERE / "revision_exploratory.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for step, v in out.items():
        for k, r in v.items():
            print(f"{step} {k}: {r['mean_diff']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] {r['b_gt_a']}/10 p {r['t_p']:.2g}")


if __name__ == "__main__":
    main()
