"""Check every cell of the results tables in paper/ROBUSTNESS_PAPER_DRAFT.md (§4.1–4.7) against the audited summary JSONs.

Unlike check_robustness_paper_numbers.py (which only checks that a number occurs somewhere in a source), this checks that
each number sits in the right row and column: table row -> (arm, condition, statistic) -> value in
experiments/robust/{summary_prefix-p2__ref-p2_full,phase3/summary,phase4a/summary,phase4abis/summary,phase4b/summary}.json.
Values printed with 3 decimals must match to 0.0005, 2 decimals to 0.005; p-values to 6% relative (they are rounded
to 2–4 significant digits). CIs for Phase 2 are recomputed from the stored paired differences (t, df = 9), as in the report.
Exits 1 on any mismatch.

Usage:
    .venv/Scripts/python.exe paper/check_robustness_paper_tables.py
"""

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "experiments" / "robust"
NUM = re.compile(r"[+\-]?\d+(?:\.\d+)?(?:e[+\-]?\d+)?")
fails, n_checks = [], 0


def load(p):
    return json.loads((R / p).read_text(encoding="utf-8"))


def table(text, heading):
    """Rows (list of cell lists) of the first markdown table after `heading`."""
    part = text.split(heading, 1)[1]
    rows = []
    for ln in part.splitlines()[1:]:
        if ln.startswith("|"):
            rows.append([c.strip() for c in ln.strip().strip("|").split("|")])
        elif rows:
            break
    return rows[2:]  # drop header and separator


def nums(cell):
    return [x for x in NUM.findall(cell.replace("−", "-"))]


def check(label, cell, expected, kind="num"):
    """Compare the numbers in one cell with expected values (list)."""
    global n_checks
    got = nums(cell)
    if kind == "count":
        ok = cell.replace(" ", "") == expected
    elif len(got) != len(expected):
        ok = False
    else:
        ok = True
        for g, e in zip(got, expected):
            if kind == "p":
                ok &= math.isclose(float(g), e, rel_tol=0.06) or (e > 0.995 and float(g) == 1.0)
            else:
                dec = len(g.split(".")[1]) if "." in g else 0
                ok &= abs(float(g) - e) <= 0.5 * 10 ** -dec + 1e-9
    n_checks += 1
    if not ok:
        fails.append(f"{label}: paper '{cell}' vs expected {expected}")


def ci95(d):
    d = np.asarray(d, float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    return [d.mean() - h, d.mean() + h]


def ms(x):
    return [x["mean"], x["sd"]]


def check_robustness_tables(text, h) -> None:
    """The §4.1-4.7 checks; h maps each draft section number ("4.1"...) to its heading in `text`."""

    # 4.1 Phase 2, validation-selected threshold
    p2 = load("summary_prefix-p2__ref-p2_full.json")
    pair = {r["condition"]: r for r in p2["p2_dropout"]["paired_vs_p2_full"]}
    keys = {"full": "full", "post optical missing (`no_o2`)": "no_o2", "copy-forward (`o2_copy`)": "o2_copy", "no optical (`no_opt`)": "no_opt"}
    for row in table(text, h["4.1"]):
        c = keys[row[0]]
        A, B, r = p2["p2_full"]["conditions"][c]["at_selected"]["f1"], p2["p2_dropout"]["conditions"][c]["at_selected"]["f1"], pair[c]
        check(f"4.1 {c} A", row[1], ms(A)), check(f"4.1 {c} B", row[2], ms(B)), check(f"4.1 {c} Δ", row[3], [r["mean_diff"]])
        check(f"4.1 {c} CI", row[4], ci95(r["diffs"]))
        check(f"4.1 {c} B>A", row[5], f"{sum(x > 0 for x in r['diffs'])}/10", "count")
        check(f"4.1 {c} Holm", row[6], [r["holm_p"]], "p")

    # 4.2 Phase 3, frozen clean threshold
    p3 = load("phase3/summary.json")
    rows3 = p3["paired"]["primary_clean"]["rows"]
    for row in table(text, h["4.2"]):
        c = row[0]
        check(f"4.2 {c} A", row[1], ms(p3["arms"]["p2_full"]["clean"][c]["f1"]))
        check(f"4.2 {c} B", row[2], ms(p3["arms"]["p2_dropout"]["clean"][c]["f1"]))
        if c != "clean":
            r = rows3[c]
            check(f"4.2 {c} Δ", row[3], [r["mean_diff"]]), check(f"4.2 {c} Holm", row[4], [r["holm_p"]], "p")
            check(f"4.2 {c} B>A", row[5], f"{r['b_gt_a']}/10", "count")

    # 4.3 Step 4A, zero-fill vs cloud
    p4a = load("phase4a/summary.json")
    rows4a = p4a["paired"]["primary_zero_vs_cloud"]["rows"]
    for row in table(text, h["4.3"]):
        arm = {"A": "p2_full", "B": "p2_dropout"}[row[0]]
        L = row[1].rstrip("%")
        f = p4a["arms"][arm]["clean"]
        check(f"4.3 {arm} {L} cloud", row[2], [f[f"c{L}"]["f1"]["mean"]]), check(f"4.3 {arm} {L} zero", row[3], [f[f"z{L}"]["f1"]["mean"]])
        check(f"4.3 {arm} {L} Δ", row[4], [rows4a[f"{arm}:z{L}-c{L}"]["mean_diff"]])
        check(f"4.3 {arm} {L} recovery", row[5], ms(p4a["recovery"][arm][f"z{L}"]))

    # 4.4 Step 4A-bis, B at 30%
    ab = load("phase4abis/summary.json")
    fam = ab["families"]["primary_B_c30_imperfect_minus_oracle"]
    Bf = ab["arms"]["p2_dropout"]["clean"]
    cloud, oracle = np.array(Bf["c30"]["f1"]["values"]), np.array(Bf["z30"]["f1"]["values"])
    for row in table(text, h["4.4"]):
        if row[0].startswith("oracle"):
            check("4.4 oracle F1", row[3], [Bf["z30"]["f1"]["mean"]])
            continue
        m = row[0]
        es = ab["error_stats"][f"c30__{m}"]
        check(f"4.4 {m} missed", row[1], [es["missed_cloud_frac"]]), check(f"4.4 {m} over", row[2], [es["overmasked_clear_frac"]])
        check(f"4.4 {m} F1", row[3], [Bf[f"c30__{m}"]["f1"]["mean"]])
        check(f"4.4 {m} Δ", row[4], [fam[m]["mean_diff"]]), check(f"4.4 {m} CI", row[5], fam[m]["ci95"])
        ret = ((np.array(Bf[f"c30__{m}"]["f1"]["values"]) - cloud) / (oracle - cloud)).mean()
        check(f"4.4 {m} retained", row[6], [ret])

    # 4.5 Step 4B
    s4 = load("phase4b/summary.json")
    names = {"c30, erode4": "c30__erode4", "c30, dilate4": "c30__dilate4", "c30, miss25": "c30__miss25"}
    for row in table(text, h["4.5"]):
        c = names.get(row[0], row[0])
        r = s4["primary"][c]
        check(f"4.5 {c} B", row[1], ms(s4["arms"]["p2_dropout"][c])), check(f"4.5 {c} C", row[2], ms(s4["arms"]["p4b_spatial"][c]))
        check(f"4.5 {c} Δ", row[3], [r["mean_diff"]]), check(f"4.5 {c} CI", row[4], r["ci95"])
        check(f"4.5 {c} C>B", row[5], f"{r['b_gt_a']}/10", "count"), check(f"4.5 {c} Holm", row[6], [r["holm_p"]], "p")

    # 4.6 Step 5S, B at 30%, P vs P dilated by r*
    s5s = load("phase5s/summary.json")
    rs, B5s = s5s["radius_selection"]["r_star"], s5s["arms"]["p2_dropout"]
    for row in table(text, h["4.6"]):
        k = f"c30__{row[0]}"
        r = s5s["primary"][k]
        check(f"4.6 {k} P", row[1], ms(B5s[f"{k}__0"])), check(f"4.6 {k} P r*", row[2], ms(B5s[f"{k}__{rs}"]))
        check(f"4.6 {k} Δ", row[3], [r["mean_diff"]]), check(f"4.6 {k} CI", row[4], r["ci95"])
        check(f"4.6 {k} Δ>0", row[5], f"{r['b_gt_a']}/10", "count"), check(f"4.6 {k} Holm", row[6], [r["holm_p"]], "p")
        check(f"4.6 {k} vs oracle", row[7], [s5s["secondary_B_vs_oracle"][k]["mean_diff"]])

    # 4.7 Step 5, primary family and descriptive detector table
    s5 = load("phase5/summary.json")
    for row in table(text, h["4.7"]):
        k = row[0].replace("−", "-")
        r = s5["primary"][k]
        check(f"4.7 {k} Δ", row[1], [r["mean_diff"]]), check(f"4.7 {k} CI", row[2], r["ci95"])
        check(f"4.7 {k} Δ>0", row[3], f"{r['b_gt_a']}/10", "count"), check(f"4.7 {k} Holm", row[4], [r["holm_p"]], "p")
        check(f"4.7 {k} label", row[5].replace(" ", ""), r["label"].replace(" ", ""), "count")
    B5 = s5["arms"]["p2_dropout"]
    for row in table(text, "**Detector masks on the test composites"):
        L, d = row[0], s5["detector_test"][row[0]]
        check(f"4.7 {L} rb95 recall", row[1], [d["rb95"]["recall"]]), check(f"4.7 {L} rb95 over", row[2], [d["rb95"]["overmasked_clear"]])
        check(f"4.7 {L} bal recall", row[3], [d["bal"]["recall"]]), check(f"4.7 {L} bal over", row[4], [d["bal"]["overmasked_clear"]])
        for i, c in enumerate(("cloud", "bal", "rb95", "rb99", "oracle"), 5):
            check(f"4.7 {L} B {c}", row[i], [B5[f"{L}__{c}"]["f1"]["mean"]])
        check(f"4.7 {L} retained", row[10], ms(s5["retained_gain"]["p2_dropout"][L]["rb95"]))



def main() -> None:
    text = (ROOT / "paper" / "ROBUSTNESS_PAPER_DRAFT.md").read_text(encoding="utf-8")
    check_robustness_tables(text, {k: f"### {k}" for k in ("4.1", "4.2", "4.3", "4.4", "4.5", "4.6", "4.7")})
    print(f"{n_checks} table cells checked, {len(fails)} mismatches")
    for f in fails:
        print("  MISMATCH", f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
