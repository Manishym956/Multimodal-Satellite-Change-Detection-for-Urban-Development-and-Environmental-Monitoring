"""Phase 3 analysis (fixed in PHASE3_PREREGISTRATION.md before eval_phase3.py ran).

Reads experiments/robust/phase3/runs/*.{json,npz}; writes phase3/summary.json, phase3/tables.md, phase3/curves.png.

Thresholds: "clean" = each checkpoint's clean-validation threshold (PRIMARY), "level" = per-condition
validation argmax (secondary, adaptation), "0.5" (secondary).
Primary family: paired B - A test F1 at the clean threshold for c10, c30, c60, missing; paired t-test,
Holm over these 4; Wilcoxon, 95% CI and d_z reported alongside. Everything else is secondary/descriptive.

Usage:
    .venv/Scripts/python.exe analyze_phase3.py
"""

import json
import math

import numpy as np
import torch
from scipy import stats

from src.phase3.corruption import PHASE3
from src.training.train_robust import rates_at

ARMS = {"p2_full": "A full-only", "p2_dropout": "B dropout"}
SEEDS = list(range(1, 11))
CONDITIONS = ("clean", "c10", "c30", "c60", "missing", "c100", "haze")
LEVELS = ("c10", "c30", "c60")
PRIMARY = ("c10", "c30", "c60", "missing")
METRICS = ("f1", "precision", "recall", "iou", "false_alarm_rate")


def holm(pvals: list[float]) -> list[float]:
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(np.argsort(pvals)):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def fmt_p(p: float) -> str:
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def describe(values: list[float]) -> dict:
    return {"values": values, "n": len(values), "mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))}


def paired(a: list[float], b: list[float]) -> dict:
    d = [y - x for x, y in zip(a, b)]
    sd = float(np.std(d, ddof=1))
    half = stats.t.ppf(0.975, len(d) - 1) * sd / math.sqrt(len(d))
    return {"diffs": d, "mean_diff": float(np.mean(d)), "sd_diff": sd, "ci95": [float(np.mean(d) - half), float(np.mean(d) + half)],
            "dz": float(np.mean(d) / sd) if sd else float("nan"), "b_gt_a": sum(x > 0 for x in d),
            "t_p": float(stats.ttest_rel(b, a).pvalue) if sd else 1.0, "wilcoxon_p": float(stats.wilcoxon(b, a).pvalue) if any(d) else 1.0}


def main() -> None:
    runs = {}
    for arm in ARMS:
        for s in SEEDS:
            doc = json.loads((PHASE3 / "runs" / f"{arm}_seed{s}.json").read_text(encoding="utf-8"))
            assert all(doc["anchors"].values()), doc["run"]
            npz = np.load(PHASE3 / "runs" / f"{arm}_seed{s}.npz")
            runs[(arm, s)] = (doc, {k: torch.from_numpy(npz[k]) for k in npz.files})
    hashes = {json.dumps(d["source_sha256"], sort_keys=True) for d, _ in runs.values()}
    assert len(hashes) == 1, "runs used different source versions"

    def k_of(doc, c, thr):
        return {"clean": round(doc["tau_clean"] * 100), "level": round(doc["tau_level_val"][c] * 100), "0.5": 50}[thr]

    def metric(arm, s, c, thr, m, table="test"):
        doc, arr = runs[(arm, s)]
        return rates_at(arr[f"{table}__{c}"], k_of(doc, c, thr))[m] or 0.0

    out = {"arms": {}, "paired": {}, "regions": {}, "source_sha256": json.loads(hashes.pop())}
    for arm in ARMS:
        out["arms"][arm] = {thr: {c: {m: describe([metric(arm, s, c, thr, m) for s in SEEDS]) for m in METRICS}
                                  for c in CONDITIONS} for thr in ("clean", "level", "0.5")}
        for thr in ("clean", "level", "0.5"):
            for c in CONDITIONS:
                drop = [metric(arm, s, c, thr, "f1") - metric(arm, s, "clean", thr, "f1") for s in SEEDS]
                out["arms"][arm][thr][c]["f1_change_from_clean"] = describe(drop)
        out["arms"][arm]["tau_level_val"] = {c: [runs[(arm, s)][0]["tau_level_val"][c] for s in SEEDS] for c in CONDITIONS}
        out["arms"][arm]["tau_clean"] = [runs[(arm, s)][0]["tau_clean"] for s in SEEDS]

    families = {"primary_clean": ("clean", PRIMARY), "secondary_clean_extra": ("clean", ("c100", "haze")),
                "secondary_level": ("level", PRIMARY + ("c100", "haze")), "secondary_0.5": ("0.5", PRIMARY + ("c100", "haze"))}
    for fam, (thr, conds) in families.items():
        rows = {c: paired([metric("p2_full", s, c, thr, "f1") for s in SEEDS], [metric("p2_dropout", s, c, thr, "f1") for s in SEEDS])
                for c in conds}
        for c, h in zip(conds, holm([rows[c]["t_p"] for c in conds])):
            rows[c]["holm_p"] = h
        for c, h in zip(conds, holm([rows[c]["wilcoxon_p"] for c in conds])):
            rows[c]["holm_wilcoxon_p"] = h
        out["paired"][fam] = {"threshold": thr, "rows": rows}

    # failure modes at the clean threshold: inside vs outside the cloud mask, corrupted vs clean input on the same pixels
    for arm in ARMS:
        out["regions"][arm] = {}
        for lv in LEVELS:
            entry = {}
            for region in ("inside", "outside"):
                for inp, key in (("corrupted", region), ("clean", f"{region}_clean")):
                    for m in ("false_alarm_rate", "recall", "precision", "f1"):
                        entry[f"{region}_{inp}_{m}"] = describe([metric(arm, s, lv, "clean", m, key) for s in SEEDS])
                    entry[f"{region}_{inp}_fp"] = describe([rates_at(runs[(arm, s)][1][f"{key}__{lv}"], k_of(runs[(arm, s)][0], lv, "clean"))["fp"] for s in SEEDS])
            share = []
            for s in SEEDS:
                doc, arr = runs[(arm, s)]
                k = k_of(doc, lv, "clean")
                fi, fo = rates_at(arr[f"inside__{lv}"], k)["fp"], rates_at(arr[f"outside__{lv}"], k)["fp"]
                share.append(fi / (fi + fo) if fi + fo else 0.0)
            entry["share_of_fp_inside_mask"] = describe(share)
            out["regions"][arm][lv] = entry
    (PHASE3 / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (PHASE3 / "tables.md").write_text(tables(out), encoding="utf-8")
    plot(out)
    print((PHASE3 / "tables.md").read_text(encoding="utf-8"))


def ms(e: dict) -> str:
    return f"{e['mean']:.3f} ± {e['sd']:.3f}"


def tables(out: dict) -> str:
    L = []
    labels = {"clean": "clean-validation threshold (PRIMARY)", "level": "per-condition validation threshold (secondary, adaptation)",
              "0.5": "threshold 0.5 (secondary)"}
    for thr in ("clean", "level", "0.5"):
        L += [f"### Test metrics at the {labels[thr]}, mean ± sample SD over 10 seeds", "",
              "| arm | condition | F1 | ΔF1 vs clean | precision | recall | IoU | false-alarm rate |", "|---|---|---|---|---|---|---|---|"]
        for arm, name in ARMS.items():
            for c in CONDITIONS:
                e = out["arms"][arm][thr][c]
                L.append(f"| {name} | {c} | {ms(e['f1'])} | {e['f1_change_from_clean']['mean']:+.3f} ± {e['f1_change_from_clean']['sd']:.3f} | "
                         f"{ms(e['precision'])} | {ms(e['recall'])} | {ms(e['iou'])} | {e['false_alarm_rate']['mean']:.4f} ± {e['false_alarm_rate']['sd']:.4f} |")
        L.append("")
    L += ["### Per-seed test F1 at the clean-validation threshold (primary)", "",
          "| seed | " + " | ".join(f"{c} A | {c} B" for c in CONDITIONS) + " |", "|---|" + "---|---|" * len(CONDITIONS)]
    for i, s in enumerate(SEEDS):
        L.append(f"| {s} | " + " | ".join(f"{out['arms'][a]['clean'][c]['f1']['values'][i]:.3f}" for c in CONDITIONS for a in ARMS) + " |")
    L.append("")
    names = {"primary_clean": "PRIMARY family (clean threshold; Holm over c10, c30, c60, missing)",
             "secondary_clean_extra": "secondary: c100 and haze at the clean threshold (own Holm family)",
             "secondary_level": "secondary: per-condition validation threshold (own Holm family)",
             "secondary_0.5": "secondary: threshold 0.5 (own Holm family)"}
    for fam, block in out["paired"].items():
        L += [f"### Paired B − A test F1, {names[fam]}", "",
              "| condition | mean Δ | SD Δ | 95% CI | d_z | B > A | t p | Holm p (t) | Wilcoxon p | Holm p (W) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for c, r in block["rows"].items():
            L.append(f"| {c} | {r['mean_diff']:+.3f} | {r['sd_diff']:.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['dz']:+.2f} | "
                     f"{r['b_gt_a']}/10 | {fmt_p(r['t_p'])} | {fmt_p(r['holm_p'])} | {fmt_p(r['wilcoxon_p'])} | {fmt_p(r['holm_wilcoxon_p'])} |")
        L.append("")
    L += ["### Failure modes at the clean threshold: pixels inside vs outside the cloud mask (same pixels, clean vs corrupted input)", "",
          "| arm | level | FAR inside: clean → corrupted | FAR outside: clean → corrupted | recall inside: clean → corrupted | recall outside: clean → corrupted | share of FP inside mask |",
          "|---|---|---|---|---|---|---|"]
    for arm, name in ARMS.items():
        for lv in LEVELS:
            e = out["regions"][arm][lv]
            L.append(f"| {name} | {lv} | {e['inside_clean_false_alarm_rate']['mean']:.4f} → {e['inside_corrupted_false_alarm_rate']['mean']:.4f} | "
                     f"{e['outside_clean_false_alarm_rate']['mean']:.4f} → {e['outside_corrupted_false_alarm_rate']['mean']:.4f} | "
                     f"{e['inside_clean_recall']['mean']:.3f} → {e['inside_corrupted_recall']['mean']:.3f} | "
                     f"{e['outside_clean_recall']['mean']:.3f} → {e['outside_corrupted_recall']['mean']:.3f} | {ms(e['share_of_fp_inside_mask'])} |")
    L.append("\n(Region values are means over 10 seeds. FAR = FP / (FP + TN). The mask fraction equals the level, so with clean input\n"
             "the share of FP inside the mask would be close to the level if false alarms were spread evenly.)")
    return "\n".join(L) + "\n"


def plot(out: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"p2_full": "#2a78d6", "p2_dropout": "#eb6834"}
    markers = {"p2_full": "o", "p2_dropout": "s"}
    ladder = ("clean", "c10", "c30", "c60", "missing")
    xlab = ("clean", "10 %", "30 %", "60 %", "missing\n(zero-fill)")
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (m, title) in zip(axes, (("f1", "F1"), ("precision", "Precision"), ("recall", "Recall"), ("false_alarm_rate", "False-alarm rate"))):
        for arm, name in ARMS.items():
            mean = [out["arms"][arm]["clean"][c][m]["mean"] for c in ladder]
            sd = [out["arms"][arm]["clean"][c][m]["sd"] for c in ladder]
            ax.errorbar(range(len(ladder)), mean, yerr=sd, color=colors[arm], marker=markers[arm], markersize=6, linewidth=2,
                        capsize=3, label=name)
        ax.set_xticks(range(len(ladder)), xlab, fontsize=8)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(axis="y", color="#e0e0e0", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("test value (mean ± SD, 10 seeds)", fontsize=8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Post-change optical corruption, frozen clean-validation threshold (simulated clouds, real cloud shapes)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(PHASE3 / "curves.png", dpi=150)


if __name__ == "__main__":
    assert np.allclose(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    main()
