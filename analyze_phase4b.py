"""Step 4B analysis (fixed in PHASE4B_PREREGISTRATION.md before any p4b_spatial run).

C = p4b_spatial (spatial partial zero-fill added to dropout training), B = p2_dropout (Phase 2), paired by seed 1-10.
Primary (frozen clean-validation threshold): paired C - B test F1 at clean, z10, z30, z60, c30__erode4, c30__dilate4,
c30__miss25; paired t-test, Holm over these 7; claim "C better at X" only if Holm p < 0.05 and Δ > 0; the clean
difference is reported as a cost/benefit whatever its sign.
Secondary (descriptive, own Holm families): C - B for all other 4A-bis conditions (cloud no-fill, other mask errors);
C - B on the Phase 2 conditions (full, no_o2, no_opt, o2_copy) at each run's validation-selected threshold from metrics.json.
Pre-specified labels per primary condition: "C better" (Holm p < 0.05, Δ > 0), "C worse" (Holm p < 0.05, Δ < 0), otherwise
"no detectable difference"; the clean difference is labelled a "substantial clean regression" if mean Δ < -0.02 F1,
whatever its p-value.
Descriptive (no tests): precision, recall, IoU and false-alarm rate for the primary conditions; metrics inside / outside the
true cloud mask M at 30% (B from the 4A-bis exploratory addendum runs, C from eval_phase4b.py), same partition and threshold.

Writes experiments/robust/phase4b/{summary.json, tables.md, curves.png}.

Usage:
    .venv/Scripts/python.exe analyze_phase4b.py
"""

import json
from pathlib import Path

import numpy as np
import torch

from analyze_phase3 import describe, fmt_p, holm, ms, paired
from eval_phase4abis import CONDITIONS
from src.phase3.mask_errors import VARIANTS
from src.training.train_robust import rates_at

ROOT = Path(__file__).resolve().parent / "experiments" / "robust"
SEEDS = list(range(1, 11))
ARMS = {"p2_dropout": ("phase4abis", "B dropout"), "p4b_spatial": ("phase4b", "C spatial dropout")}
PRIMARY = ("clean", "z10", "z30", "z60", "c30__erode4", "c30__dilate4", "c30__miss25")
P2_CONDITIONS = ("full", "no_o2", "no_opt", "o2_copy")


def main() -> None:
    runs = {}
    for arm, (folder, _) in ARMS.items():
        for s in SEEDS:
            doc = json.loads((ROOT / folder / "runs" / f"{arm}_seed{s}.json").read_text(encoding="utf-8"))
            assert all(doc["anchors"].values()), doc["run"]
            npz = np.load(ROOT / folder / "runs" / f"{arm}_seed{s}.npz")
            runs[(arm, s)] = (doc, {k: torch.from_numpy(npz[k]) for k in npz.files},
                              json.loads((ROOT / f"{arm}_seed{s}" / "metrics.json").read_text(encoding="utf-8")))
    c_args = {json.dumps({k: v for k, v in runs[("p4b_spatial", s)][2]["args"].items() if k != "seed"}, sort_keys=True) for s in SEEDS}
    assert len(c_args) == 1, "p4b_spatial seeds differ in arguments"

    def f1(arm, c, s):
        doc, arr, _ = runs[(arm, s)]
        return rates_at(arr[f"test__{c}"], round(doc["tau_clean"] * 100))["f1"] or 0.0

    out = {"arms": {}, "primary": {}, "secondary_other_conditions": {}, "secondary_phase2_conditions": {}}
    for arm in ARMS:
        out["arms"][arm] = {c: describe([f1(arm, c, s) for s in SEEDS]) for c in CONDITIONS}
        out["arms"][arm]["phase2"] = {c: {m: describe([runs[(arm, s)][2]["test"][c]["at_selected"][m] for s in SEEDS])
                                          for m in ("f1", "precision", "recall", "iou")} for c in P2_CONDITIONS}
        out["arms"][arm]["selected_steps"] = [runs[(arm, s)][2]["selected_step"] for s in SEEDS]

    def family(conds, getter):
        rows = {c: paired(getter("p2_dropout", c), getter("p4b_spatial", c)) for c in conds}
        for c, h in zip(rows, holm([r["t_p"] for r in rows.values()])):
            rows[c]["holm_p"] = h
        for c, h in zip(rows, holm([r["wilcoxon_p"] for r in rows.values()])):
            rows[c]["holm_wilcoxon_p"] = h
        return rows

    getter = lambda arm, c: [f1(arm, c, s) for s in SEEDS]
    out["primary"] = family(PRIMARY, getter)
    out["secondary_other_conditions"] = family([c for c in CONDITIONS if c not in PRIMARY], getter)
    out["secondary_phase2_conditions"] = family(P2_CONDITIONS, lambda arm, c: [runs[(arm, s)][2]["test"][c]["at_selected"]["f1"] for s in SEEDS])
    for c, r in out["primary"].items():
        r["label"] = ("C better" if r["holm_p"] < 0.05 and r["mean_diff"] > 0 else
                      "C worse" if r["holm_p"] < 0.05 and r["mean_diff"] < 0 else "no detectable difference")
    out["substantial_clean_regression"] = bool(out["primary"]["clean"]["mean_diff"] < -0.02)

    def rate(arm, s, key, m):
        doc, arr, _ = runs[(arm, s)]
        return rates_at(arr[key], round(doc["tau_clean"] * 100))[m] or 0.0

    out["metrics"] = {arm: {c: {m: describe([rate(arm, s, f"test__{c}", m) for s in SEEDS])
                                for m in ("f1", "precision", "recall", "iou", "false_alarm_rate")} for c in PRIMARY} for arm in ARMS}
    io_conds = ("clean@c30", "c30", "z30", *(f"c30__{v}" for v in VARIANTS))
    io = {"p2_dropout": [np.load(ROOT / "phase4abis" / "addendum_inside_outside" / "runs" / f"p2_dropout_seed{s}.npz") for s in SEEDS],
          "p4b_spatial": [np.load(ROOT / "phase4b" / "runs" / f"p4b_spatial_seed{s}.npz") for s in SEEDS]}
    prefix = {"p2_dropout": "", "p4b_spatial": "io_"}
    out["inside_outside"] = {arm: {c: {f"{part}_{m}": describe([rates_at(torch.from_numpy(a[f"{prefix[arm]}{part}__{c}"]),
                                                                          round(runs[(arm, s)][0]["tau_clean"] * 100))[m] or 0.0
                                                                 for a, s in zip(io[arm], SEEDS)])
                                       for part in ("inside", "outside") for m in ("f1", "recall", "false_alarm_rate")}
                                   for c in io_conds} for arm in ARMS}
    (ROOT / "phase4b" / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (ROOT / "phase4b" / "tables.md").write_text(tables(out), encoding="utf-8")
    plot(out)
    print((ROOT / "phase4b" / "tables.md").read_text(encoding="utf-8"))


def family_table(rows: dict, title: str, out: dict, phase2: bool = False) -> list[str]:
    L = [title, "", "| condition | B F1 | C F1 | mean Δ (C−B) | 95% CI | d_z | C > B | t p | Holm p (t) | Wilcoxon p | Holm p (W) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, r in rows.items():
        b = out["arms"]["p2_dropout"]["phase2"][c]["f1"] if phase2 else out["arms"]["p2_dropout"][c]
        cc = out["arms"]["p4b_spatial"]["phase2"][c]["f1"] if phase2 else out["arms"]["p4b_spatial"][c]
        L.append(f"| {c} | {ms(b)} | {ms(cc)} | {r['mean_diff']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['dz']:+.2f} | "
                 f"{r['b_gt_a']}/10 | {fmt_p(r['t_p'])} | {fmt_p(r['holm_p'])} | {fmt_p(r['wilcoxon_p'])} | {fmt_p(r['holm_wilcoxon_p'])} |")
    return L + [""]


def tables(out: dict) -> str:
    L = family_table(out["primary"], "### PRIMARY: paired C − B test F1, frozen clean threshold, Holm over 7", out)
    L += ["Pre-specified labels: " + "; ".join(f"{c}: **{r['label']}**" for c, r in out["primary"].items())
          + f". Substantial clean regression (mean Δ < −0.02): **{'yes' if out['substantial_clean_regression'] else 'no'}**.", ""]
    L += ["### Primary conditions: F1 / precision / recall / IoU / false-alarm rate at the frozen clean threshold (mean ± SD)", "",
          "| arm | condition | F1 | precision | recall | IoU | false-alarm rate |", "|---|---|---|---|---|---|---|"]
    for arm, (_, name) in ARMS.items():
        for c in PRIMARY:
            e = out["metrics"][arm][c]
            L.append(f"| {name} | {c} | " + " | ".join(ms(e[m]) for m in ("f1", "precision", "recall", "iou", "false_alarm_rate")) + " |")
    L += ["", "### Descriptive: 30% cloud, inside / outside the true cloud mask M, frozen clean threshold (mean ± SD; no tests)", "",
          "| arm | condition | F1 in M | recall in M | FAR in M | F1 outside M | recall outside M | FAR outside M |", "|---|---|---|---|---|---|---|---|"]
    for arm, (_, name) in ARMS.items():
        for c, e in out["inside_outside"][arm].items():
            L.append(f"| {name} | {c} | " + " | ".join(ms(e[f"{p}_{m}"]) for p in ("inside", "outside") for m in ("f1", "recall", "false_alarm_rate")) + " |")
    L.append("")
    L += family_table(out["secondary_other_conditions"], "### Secondary: other 4A-bis conditions, frozen clean threshold (own Holm family)", out)
    L += family_table(out["secondary_phase2_conditions"], "### Secondary: Phase 2 conditions at each run's validation-selected threshold (own Holm family)", out, True)
    L += ["### Phase 2 conditions, precision / recall / IoU at the validation-selected threshold (mean ± SD)", "",
          "| arm | condition | precision | recall | IoU |", "|---|---|---|---|---|"]
    for arm, (_, name) in ARMS.items():
        for c in P2_CONDITIONS:
            e = out["arms"][arm]["phase2"][c]
            L.append(f"| {name} | {c} | {ms(e['precision'])} | {ms(e['recall'])} | {ms(e['iou'])} |")
    L += ["", "Selected checkpoint steps: " + "; ".join(f"{ARMS[a][1]} {out['arms'][a]['selected_steps']}" for a in ARMS)]
    L += ["", "### Per-seed test F1 at the frozen clean threshold, primary conditions", "",
          "| seed | " + " | ".join(f"{c} B | {c} C" for c in PRIMARY) + " |", "|---|" + "---|---|" * len(PRIMARY)]
    for i, s in enumerate(SEEDS):
        L.append(f"| {s} | " + " | ".join(f"{out['arms'][a][c]['values'][i]:.3f}" for c in PRIMARY for a in ARMS) + " |")
    return "\n".join(L) + "\n"


def plot(out: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"p2_dropout": "#eb6834", "p4b_spatial": "#1baf7a"}
    groups = (("cloud, no fill", ("clean", "c10", "c30", "c60")), ("oracle fill", ("clean", "z10", "z30", "z60")),
              ("30% cloud, imperfect fill", ("z30", *(f"c30__{v}" for v in VARIANTS))))
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True, gridspec_kw={"width_ratios": [1, 1, 2.2]})
    for ax, (title, conds) in zip(axes, groups):
        for arm, (_, name) in ARMS.items():
            mean = [out["arms"][arm][c]["mean"] for c in conds]
            sd = [out["arms"][arm][c]["sd"] for c in conds]
            ax.errorbar(range(len(conds)), mean, yerr=sd, color=colors[arm], marker="o" if arm == "p2_dropout" else "s",
                        markersize=6, linewidth=2, capsize=3, label=name)
        ax.set_xticks(range(len(conds)), [c.replace("c30__", "").replace("z30", "oracle") for c in conds], fontsize=7, rotation=30)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(axis="y", color="#e0e0e0", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("test F1, frozen clean threshold\n(mean ± SD, 10 seeds)", fontsize=8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "phase4b" / "curves.png", dpi=150)


if __name__ == "__main__":
    main()
