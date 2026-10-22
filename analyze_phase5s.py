"""Step 5S analysis (fixed in PHASE5S_PREREGISTRATION.md before eval_phase5s.py ran).

Radius selection (VALIDATION only, B = p2_dropout): r* = the radius in RADII with the highest mean, over B's 10 seeds, of
the mean validation F1 (frozen clean threshold) across all 18 under-masking conditions (3 levels x 6 variants); ties: smaller r.
PRIMARY (B, test, c30, Holm over 6): F1(c30__v__r*) - F1(c30__v__0) for v in UNDER. Paired t-test, two-sided.
  Labels: "dilation better" (Holm p < 0.05, Δ > 0), "dilation worse" (Holm p < 0.05, Δ < 0), else "no detectable difference".
SECONDARY (own Holm families, descriptive): the same contrast for B at c10 and c60 (12); for A and for C at all levels (18 each);
B: F1(L__v__r*) - F1(zL) (cost vs oracle, 18).
DESCRIPTIVE: test F1 mean ± SD for every condition and arm (the full curve over r); retained gain at r*
(F1 - F1_cloud) / (F1_oracle - F1_cloud).
Writes experiments/robust/phase5s/{summary.json, tables.md}.

Usage:
    .venv/Scripts/python.exe analyze_phase5s.py
"""

import json

import numpy as np
import torch

from analyze_phase3 import describe, fmt_p, holm, ms, paired
from eval_phase5s import ARMS, CONDITIONS, OUT, RADII, UNDER
from src.phase3.corruption import LEVELS
from src.training.train_robust import rates_at

SEEDS = list(range(1, 11))
NAMES = {"p2_full": "A full-only", "p2_dropout": "B dropout", "p4b_spatial": "C spatial dropout"}


def main() -> None:
    runs = {}
    for arm in ARMS:
        for s in SEEDS:
            doc = json.loads((OUT / "runs" / f"{arm}_seed{s}.json").read_text(encoding="utf-8"))
            assert all(doc["anchors"].values()), doc["run"]
            npz = np.load(OUT / "runs" / f"{arm}_seed{s}.npz")
            runs[(arm, s)] = (doc, {k: torch.from_numpy(npz[k]) for k in npz.files})
    assert len({json.dumps(d["source_sha256"], sort_keys=True) for d, _ in runs.values()}) == 1

    def f1(arm, c, split="test"):
        return [rates_at(runs[(arm, s)][1][f"{split}__{c}"], round(runs[(arm, s)][0]["tau_clean"] * 100))["f1"] or 0.0 for s in SEEDS]

    under = [(lv, v) for lv in LEVELS for v in UNDER]
    val_score = {r: float(np.mean([np.mean(f1("p2_dropout", f"{lv}__{v}__{r}", "val")) for lv, v in under])) for r in RADII}
    best = max(val_score.values())
    r_star = min(r for r in RADII if val_score[r] == best)

    def family(pairs):
        rows = {k: paired(a, b) for k, (a, b) in pairs.items()}
        for k, h in zip(rows, holm([r["t_p"] for r in rows.values()])):
            rows[k]["holm_p"] = h
        for k, h in zip(rows, holm([r["wilcoxon_p"] for r in rows.values()])):
            rows[k]["holm_wilcoxon_p"] = h
        return rows

    dil = lambda arm, lvs: family({f"{lv}__{v}": (f1(arm, f"{lv}__{v}__0"), f1(arm, f"{lv}__{v}__{r_star}")) for lv in lvs for v in UNDER})
    out = {"radius_selection": {"rule": "max mean validation F1 of B over 18 under-masking conditions; ties smaller r",
                                "val_score_by_radius": val_score, "r_star": r_star}}
    out["arms"] = {arm: {c: describe(f1(arm, c)) for c in CONDITIONS} for arm in ARMS}
    out["primary"] = dil("p2_dropout", ("c30",))
    for r in out["primary"].values():
        r["label"] = ("dilation better" if r["holm_p"] < 0.05 and r["mean_diff"] > 0 else
                      "dilation worse" if r["holm_p"] < 0.05 and r["mean_diff"] < 0 else "no detectable difference")
    out["secondary_B_c10_c60"] = dil("p2_dropout", ("c10", "c60"))
    out["secondary_A"] = dil("p2_full", tuple(LEVELS))
    out["secondary_C"] = dil("p4b_spatial", tuple(LEVELS))
    out["secondary_B_vs_oracle"] = family({f"{lv}__{v}": (f1("p2_dropout", f"z{lv[1:]}"), f1("p2_dropout", f"{lv}__{v}__{r_star}"))
                                           for lv, v in under})

    def retained(arm, lv, v, r):
        c, o, x = (np.array(f1(arm, k)) for k in (lv, f"z{lv[1:]}", f"{lv}__{v}__{r}"))
        return describe(list((x - c) / (o - c)))

    out["retained_gain"] = {arm: {f"{lv}__{v}": {"r0": retained(arm, lv, v, 0), "r_star": retained(arm, lv, v, r_star)} for lv, v in under}
                            for arm in ARMS}
    (OUT / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (OUT / "tables.md").write_text(tables(out), encoding="utf-8")
    print((OUT / "tables.md").read_text(encoding="utf-8"))


def family_table(rows, title):
    L = [title, "", "| condition | mean Δ | 95% CI | d_z | Δ > 0 | t p | Holm p (t) | Wilcoxon p | Holm p (W) |", "|---|---|---|---|---|---|---|---|---|"]
    for k, r in rows.items():
        L.append(f"| {k} | {r['mean_diff']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['dz']:+.2f} | {r['b_gt_a']}/10 | "
                 f"{fmt_p(r['t_p'])} | {fmt_p(r['holm_p'])} | {fmt_p(r['wilcoxon_p'])} | {fmt_p(r['holm_wilcoxon_p'])} |"
                 + (f" **{r['label']}**" if "label" in r else ""))
    return L + [""]


def tables(out):
    rs = out["radius_selection"]
    L = [f"Validation-selected radius r* = **{rs['r_star']}** px (B, mean validation F1 over 18 under-masking conditions: "
         + ", ".join(f"r={r}: {v:.3f}" for r, v in rs["val_score_by_radius"].items()) + ")", ""]
    L += family_table(out["primary"], f"### PRIMARY: B dropout, c30, F1(P dilated by r*) − F1(P), test, Holm over 6")
    for key, title in (("secondary_B_c10_c60", "B at c10 and c60"), ("secondary_A", "A full-only"), ("secondary_C", "C spatial dropout"),
                       ("secondary_B_vs_oracle", "B, F1(P dilated by r*) − F1(oracle)")):
        L += family_table(out[key], f"### Secondary: {title} (own Holm family)")
    L += ["### Test F1 vs dilation radius, frozen clean threshold (mean ± SD, 10 seeds)", ""]
    for arm in ARMS:
        L += [f"**{NAMES[arm]}**", "", "| condition | " + " | ".join(f"r={r}" for r in RADII) + " |", "|---|" + "---|" * len(RADII)]
        for lv in LEVELS:
            for v in UNDER:
                L.append(f"| {lv}__{v} | " + " | ".join(f"{out['arms'][arm][f'{lv}__{v}__{r}']['mean']:.3f}" for r in RADII) + " |")
        L.append("")
        L.append("References: " + ", ".join(f"{c} {ms(out['arms'][arm][c])}" for c in ("clean", "c10", "c30", "c60", "z10", "z30", "z60")))
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
