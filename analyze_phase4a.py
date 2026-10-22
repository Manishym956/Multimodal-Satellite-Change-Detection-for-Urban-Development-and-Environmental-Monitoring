"""Step 4A analysis (fixed in PHASE4A_PREREGISTRATION.md before eval_phase4a.py ran).

Primary (frozen clean-validation threshold): within-arm paired difference F1(zL) - F1(cL), arms A and B,
L in {10, 30, 60}; paired t-test, Holm over these 6; Wilcoxon, 95% CI, d_z alongside.
Pre-defined descriptive label: recovery = (F1_z - F1_c) / (F1_clean - F1_c) per seed; "substantial" if the mean >= 0.5.
Secondary: B - A at z10/z30/z60 (Holm over 3); inside/outside-mask false-alarm rate and recall; thresholds 0.5 and
per-condition validation argmax (own Holm families, descriptive).

Writes experiments/robust/phase4a/{summary.json, tables.md, curves.png}.

Usage:
    .venv/Scripts/python.exe analyze_phase4a.py
"""

import json

import numpy as np
import torch

from analyze_phase3 import describe, fmt_p, holm, ms, paired
from src.training.train_robust import rates_at

OUT = __import__("pathlib").Path(__file__).resolve().parent / "experiments" / "robust" / "phase4a"
ARMS = {"p2_full": "A full-only", "p2_dropout": "B dropout"}
SEEDS = list(range(1, 11))
PAIRS = {"c10": "z10", "c30": "z30", "c60": "z60"}
CONDITIONS = ("clean", *PAIRS, *PAIRS.values())
METRICS = ("f1", "precision", "recall", "iou", "false_alarm_rate")


def main() -> None:
    runs = {}
    for arm in ARMS:
        for s in SEEDS:
            doc = json.loads((OUT / "runs" / f"{arm}_seed{s}.json").read_text(encoding="utf-8"))
            assert all(doc["anchors"].values()), doc["run"]
            npz = np.load(OUT / "runs" / f"{arm}_seed{s}.npz")
            runs[(arm, s)] = (doc, {k: torch.from_numpy(npz[k]) for k in npz.files})
    hashes = {json.dumps(d["source_sha256"], sort_keys=True) for d, _ in runs.values()}
    assert len(hashes) == 1, "runs used different source versions"

    def k_of(doc, c, thr):
        return {"clean": round(doc["tau_clean"] * 100), "level": round(doc["tau_level_val"][c] * 100), "0.5": 50}[thr]

    def metric(arm, s, c, thr, m, table="test"):
        doc, arr = runs[(arm, s)]
        return rates_at(arr[f"{table}__{c}"], k_of(doc, c, thr))[m] or 0.0

    out = {"arms": {}, "paired": {}, "recovery": {}, "regions": {}, "source_sha256": json.loads(hashes.pop())}
    for arm in ARMS:
        out["arms"][arm] = {thr: {c: {m: describe([metric(arm, s, c, thr, m) for s in SEEDS]) for m in METRICS} for c in CONDITIONS}
                            for thr in ("clean", "level", "0.5")}
        out["recovery"][arm] = {}
        for c, z in PAIRS.items():
            rec = []
            for s in SEEDS:
                f_clean, f_c, f_z = (metric(arm, s, x, "clean", "f1") for x in ("clean", c, z))
                rec.append((f_z - f_c) / (f_clean - f_c))
            out["recovery"][arm][z] = {**describe(rec), "substantial": bool(np.mean(rec) >= 0.5)}

    rows = {}
    for arm in ARMS:
        for c, z in PAIRS.items():
            rows[f"{arm}:{z}-{c}"] = paired([metric(arm, s, c, "clean", "f1") for s in SEEDS], [metric(arm, s, z, "clean", "f1") for s in SEEDS])
    for key, h in zip(rows, holm([r["t_p"] for r in rows.values()])):
        rows[key]["holm_p"] = h
    for key, h in zip(rows, holm([r["wilcoxon_p"] for r in rows.values()])):
        rows[key]["holm_wilcoxon_p"] = h
    out["paired"]["primary_zero_vs_cloud"] = {"threshold": "clean", "rows": rows}
    for fam, thr in (("secondary_B_minus_A_clean", "clean"), ("secondary_B_minus_A_level", "level"), ("secondary_B_minus_A_0.5", "0.5")):
        rows = {z: paired([metric("p2_full", s, z, thr, "f1") for s in SEEDS], [metric("p2_dropout", s, z, thr, "f1") for s in SEEDS])
                for z in PAIRS.values()}
        for z, h in zip(rows, holm([r["t_p"] for r in rows.values()])):
            rows[z]["holm_p"] = h
        for z, h in zip(rows, holm([r["wilcoxon_p"] for r in rows.values()])):
            rows[z]["holm_wilcoxon_p"] = h
        out["paired"][fam] = {"threshold": thr, "rows": rows}

    for arm in ARMS:
        out["regions"][arm] = {}
        for c, z in PAIRS.items():
            e = {}
            for region in ("inside", "outside"):
                for label, (cond, table) in (("clean", (c, f"{region}_clean")), ("cloud", (c, region)), ("zero", (z, region))):
                    for m in ("false_alarm_rate", "recall"):
                        e[f"{region}_{label}_{m}"] = describe([metric(arm, s, cond, "clean", m, table) for s in SEEDS])
            out["regions"][arm][z] = e
    (OUT / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (OUT / "tables.md").write_text(tables(out), encoding="utf-8")
    plot(out)
    print((OUT / "tables.md").read_text(encoding="utf-8"))


def tables(out: dict) -> str:
    L = ["### Paired F1(zero-fill under mask) − F1(cloud), same checkpoint, frozen clean threshold (PRIMARY; Holm over 6)", "",
         "| arm | level | F1 cloud | F1 zero-fill | mean Δ | 95% CI | d_z | z > c | t p | Holm p (t) | Wilcoxon p | Holm p (W) | recovery (mean ± SD) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, r in out["paired"]["primary_zero_vs_cloud"]["rows"].items():
        arm, pair = key.split(":")
        z, c = pair.split("-")
        rec = out["recovery"][arm][z]
        L.append(f"| {ARMS[arm]} | {c[1:]}% | {ms(out['arms'][arm]['clean'][c]['f1'])} | {ms(out['arms'][arm]['clean'][z]['f1'])} | "
                 f"{r['mean_diff']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['dz']:+.2f} | {r['b_gt_a']}/10 | {fmt_p(r['t_p'])} | "
                 f"{fmt_p(r['holm_p'])} | {fmt_p(r['wilcoxon_p'])} | {fmt_p(r['holm_wilcoxon_p'])} | {ms(rec)}{' (substantial)' if rec['substantial'] else ''} |")
    L += ["", "Recovery = (F1_zero − F1_cloud) / (F1_clean − F1_cloud), per seed; 1 = back to clean, 0 = no better than cloud; "
          "'substantial' = mean ≥ 0.5 (pre-defined descriptive label).", ""]
    labels = {"clean": "frozen clean-validation threshold (primary)", "level": "per-condition validation threshold (secondary)", "0.5": "threshold 0.5 (secondary)"}
    for thr in ("clean", "level", "0.5"):
        L += [f"### Test metrics at the {labels[thr]}, mean ± sample SD over 10 seeds", "",
              "| arm | condition | F1 | precision | recall | IoU | false-alarm rate |", "|---|---|---|---|---|---|---|"]
        for arm, name in ARMS.items():
            for c in CONDITIONS:
                e = out["arms"][arm][thr][c]
                L.append(f"| {name} | {c} | {ms(e['f1'])} | {ms(e['precision'])} | {ms(e['recall'])} | {ms(e['iou'])} | "
                         f"{e['false_alarm_rate']['mean']:.4f} ± {e['false_alarm_rate']['sd']:.4f} |")
        L.append("")
    L += ["### Per-seed test F1 at the frozen clean threshold", "",
          "| seed | " + " | ".join(f"{c} A | {c} B" for c in CONDITIONS) + " |", "|---|" + "---|---|" * len(CONDITIONS)]
    for i, s in enumerate(SEEDS):
        L.append(f"| {s} | " + " | ".join(f"{out['arms'][a]['clean'][c]['f1']['values'][i]:.3f}" for c in CONDITIONS for a in ARMS) + " |")
    L.append("")
    names = {"secondary_B_minus_A_clean": "frozen clean threshold", "secondary_B_minus_A_level": "per-condition validation threshold",
             "secondary_B_minus_A_0.5": "threshold 0.5"}
    for fam, name in names.items():
        L += [f"### Secondary: paired B − A under zero-fill, {name} (own Holm family over 3)", "",
              "| condition | mean Δ | SD Δ | 95% CI | d_z | B > A | t p | Holm p (t) | Wilcoxon p |", "|---|---|---|---|---|---|---|---|---|"]
        for z, r in out["paired"][fam]["rows"].items():
            L.append(f"| {z} | {r['mean_diff']:+.3f} | {r['sd_diff']:.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['dz']:+.2f} | "
                     f"{r['b_gt_a']}/10 | {fmt_p(r['t_p'])} | {fmt_p(r['holm_p'])} | {fmt_p(r['wilcoxon_p'])} |")
        L.append("")
    L += ["### Inside vs outside the mask, frozen clean threshold (means over 10 seeds; same pixels: clean → cloud → zero-fill)", "",
          "| arm | level | FAR inside | FAR outside | recall inside | recall outside |", "|---|---|---|---|---|---|"]
    for arm, name in ARMS.items():
        for c, z in PAIRS.items():
            e = out["regions"][arm][z]
            cell = lambda r, m: " → ".join(f"{e[f'{r}_{x}_{m}']['mean']:.{4 if m == 'false_alarm_rate' else 3}f}" for x in ("clean", "cloud", "zero"))
            L.append(f"| {name} | {c[1:]}% | {cell('inside', 'false_alarm_rate')} | {cell('outside', 'false_alarm_rate')} | "
                     f"{cell('inside', 'recall')} | {cell('outside', 'recall')} |")
    return "\n".join(L) + "\n"


def plot(out: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"p2_full": "#2a78d6", "p2_dropout": "#eb6834"}
    x = [0, 10, 30, 60]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, arm in zip(axes, ARMS):
        for series, conds, style in (("cloud (simulated)", ("clean", "c10", "c30", "c60"), "--"),
                                     ("zero-fill under oracle mask", ("clean", "z10", "z30", "z60"), "-")):
            mean = [out["arms"][arm]["clean"][c]["f1"]["mean"] for c in conds]
            sd = [out["arms"][arm]["clean"][c]["f1"]["sd"] for c in conds]
            ax.errorbar(x, mean, yerr=sd, color=colors[arm], linestyle=style, marker="o" if style == "-" else "s",
                        markersize=6, linewidth=2, capsize=3, label=series)
        ax.set_title(ARMS[arm], fontsize=10, loc="left")
        ax.set_xticks(x, ["clean", "10 %", "30 %", "60 %"])
        ax.set_xlabel("fraction of post-change optical pixels affected", fontsize=8)
        ax.grid(axis="y", color="#e0e0e0", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("test F1, frozen clean threshold (mean ± SD, 10 seeds)", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "curves.png", dpi=150)


if __name__ == "__main__":
    main()
