"""Step 4A-bis analysis (fixed in PHASE4ABIS_PREREGISTRATION.md before eval_phase4abis.py ran).

Primary (frozen clean threshold): dropout arm B at 30 % cloud, paired F1(c30 imperfect mask v) - F1(z30 oracle mask) for
the 9 variants; paired t-test, Holm over 9; mean Δ, 95% CI, d_z, Wilcoxon alongside.
Pre-defined descriptive label: retained gain = (F1_v - F1_cloud) / (F1_oracle - F1_cloud) per seed; "most of the gain
retained" if the mean >= 0.75.
Secondary (descriptive, own Holm families): B at c10 and c60; A at every level; B - A under each imperfect mask at c30;
error regions (missed cloud M\\P, over-masked clear P\\M) false-alarm rate and recall; thresholds 0.5 and per-condition validation.

Writes experiments/robust/phase4abis/{summary.json, tables.md, curves.png}.

Usage:
    .venv/Scripts/python.exe analyze_phase4abis.py
"""

import json
from pathlib import Path

import numpy as np
import torch

from analyze_phase3 import describe, fmt_p, holm, ms, paired
from src.phase3.mask_errors import VARIANTS
from src.training.train_robust import rates_at

OUT = Path(__file__).resolve().parent / "experiments" / "robust" / "phase4abis"
ARMS = {"p2_full": "A full-only", "p2_dropout": "B dropout"}
SEEDS = list(range(1, 11))
LEVELS = ("c10", "c30", "c60")


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
    manifest = json.loads((OUT / "mask_manifest.json").read_text(encoding="utf-8"))

    def metric(arm, s, c, m, thr="clean", table="test"):
        doc, arr = runs[(arm, s)]
        k = {"clean": round(doc["tau_clean"] * 100), "0.5": 50, "level": round(doc["tau_level_val"].get(c, doc["tau_clean"]) * 100)}[thr]
        return rates_at(arr[f"{table}__{c}"], k)[m] or 0.0

    f1 = lambda arm, c, thr="clean": [metric(arm, s, c, "f1", thr) for s in SEEDS]
    out = {"error_stats": {}, "arms": {}, "families": {}, "regions": {}, "source_sha256": json.loads(hashes.pop())}
    for lv in LEVELS:
        for v in VARIANTS:
            rows = [e for e in manifest if e["split"] == "test" and e["level"] == lv and e["variant"] == v]
            out["error_stats"][f"{lv}__{v}"] = {k: float(np.mean([r[k] for r in rows])) for k in ("missed_cloud_frac", "overmasked_clear_frac")}
    for arm in ARMS:
        out["arms"][arm] = {}
        for thr in ("clean", "0.5", "level"):
            out["arms"][arm][thr] = {c: {m: describe([metric(arm, s, c, m, thr) for s in SEEDS]) for m in ("f1", "precision", "recall", "false_alarm_rate")}
                                     for c in ["clean", *LEVELS, "z10", "z30", "z60", *(f"{lv}__{v}" for lv in LEVELS for v in VARIANTS)]}
        for lv in LEVELS:
            z = f"z{lv[1:]}"
            for v in VARIANTS:
                c = f"{lv}__{v}"
                ret = [(fv - fc) / (fz - fc) for fv, fc, fz in zip(f1(arm, c), f1(arm, lv), f1(arm, z))]
                out["arms"][arm]["clean"][c]["retained_gain"] = {**describe(ret), "most_retained": bool(np.mean(ret) >= 0.75)}

    def family(name, pairs):
        rows = {key: paired(a, b) for key, (a, b) in pairs.items()}
        for key, h in zip(rows, holm([r["t_p"] for r in rows.values()])):
            rows[key]["holm_p"] = h
        for key, h in zip(rows, holm([r["wilcoxon_p"] for r in rows.values()])):
            rows[key]["holm_wilcoxon_p"] = h
        out["families"][name] = rows

    family("primary_B_c30_imperfect_minus_oracle", {v: (f1("p2_dropout", "z30"), f1("p2_dropout", f"c30__{v}")) for v in VARIANTS})
    for arm in ARMS:
        for lv in LEVELS:
            if arm == "p2_dropout" and lv == "c30":
                continue
            family(f"secondary_{arm}_{lv}_imperfect_minus_oracle", {v: (f1(arm, f"z{lv[1:]}"), f1(arm, f"{lv}__{v}")) for v in VARIANTS})
    family("secondary_B_minus_A_c30", {v: (f1("p2_full", f"c30__{v}"), f1("p2_dropout", f"c30__{v}")) for v in VARIANTS})

    for arm in ARMS:
        out["regions"][arm] = {}
        for lv in LEVELS:
            for v in VARIANTS:
                c, e = f"{lv}__{v}", {}
                for r in ("missed", "over"):
                    for src, table in (("clean", f"{r}_clean"), ("input", r)):
                        for m in ("false_alarm_rate", "recall"):
                            vals = [metric(arm, s, c, m, "clean", table) for s in SEEDS]
                            e[f"{r}_{src}_{m}"] = describe(vals)
                out["regions"][arm][c] = e
    (OUT / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (OUT / "tables.md").write_text(tables(out), encoding="utf-8")
    plot(out)
    print((OUT / "tables.md").read_text(encoding="utf-8"))


def family_table(rows: dict, label: str) -> list[str]:
    L = ["| mask | mean Δ | 95% CI | d_z | > | t p | Holm p (t) | Wilcoxon p | Holm p (W) |", "|---|---|---|---|---|---|---|---|---|"]
    for key, r in rows.items():
        L.append(f"| {key} | {r['mean_diff']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['dz']:+.2f} | {r['b_gt_a']}/10 | "
                 f"{fmt_p(r['t_p'])} | {fmt_p(r['holm_p'])} | {fmt_p(r['wilcoxon_p'])} | {fmt_p(r['holm_wilcoxon_p'])} |")
    return [label, ""] + L + [""]


def tables(out: dict) -> str:
    L = ["### Test F1 at the frozen clean threshold, mean ± sample SD over 10 seeds, with mask error and retained gain", "",
         "| level | mask | missed cloud | over-masked clear | A F1 | A retained | B F1 | B retained |", "|---|---|---|---|---|---|---|---|"]
    for lv in LEVELS:
        for ref, label in ((lv, "cloud, no fill"), (f"z{lv[1:]}", "oracle fill")):
            L.append(f"| {lv[1:]}% | {label} | — | — | {ms(out['arms']['p2_full']['clean'][ref]['f1'])} | — | "
                     f"{ms(out['arms']['p2_dropout']['clean'][ref]['f1'])} | — |")
        for v in VARIANTS:
            c, es = f"{lv}__{v}", out["error_stats"][f"{lv}__{v}"]
            a, b = out["arms"]["p2_full"]["clean"][c], out["arms"]["p2_dropout"]["clean"][c]
            L.append(f"| {lv[1:]}% | {v} | {es['missed_cloud_frac']:.3f} | {es['overmasked_clear_frac']:.3f} | {ms(a['f1'])} | "
                     f"{ms(a['retained_gain'])} | {ms(b['f1'])} | {ms(b['retained_gain'])}{' (most)' if b['retained_gain']['most_retained'] else ''} |")
    L += ["", "Missed cloud = |M \\ P| / |M|; over-masked clear = |P \\ M| / |not M| (test cities, mean). Retained gain = "
          "(F1_mask − F1_cloud) / (F1_oracle − F1_cloud); 'most' = mean ≥ 0.75 (pre-defined descriptive label).", ""]
    L += family_table(out["families"]["primary_B_c30_imperfect_minus_oracle"],
                      "### PRIMARY: B dropout, 30% cloud, paired F1(imperfect mask) − F1(oracle mask), Holm over 9")
    for name, rows in out["families"].items():
        if name.startswith("secondary"):
            L += family_table(rows, f"### Secondary ({name}), own Holm family; Δ = second − first")
    L += ["### Error regions at the frozen clean threshold (means over 10 seeds; same pixels: clean input → imperfect-mask input)", "",
          "| arm | level | mask | FAR on missed cloud | recall on missed cloud | FAR on over-masked clear | recall on over-masked clear |",
          "|---|---|---|---|---|---|---|"]
    for arm, name in ARMS.items():
        for lv in LEVELS:
            for v in VARIANTS:
                e = out["regions"][arm][f"{lv}__{v}"]
                cell = lambda r, m: (f"{e[f'{r}_clean_{m}']['mean']:.3f} → {e[f'{r}_input_{m}']['mean']:.3f}"
                                     if e[f"{r}_clean_{m}"]["n"] and (r == "missed" and not v.startswith("dilate") or r == "over" and v.startswith("dilate")) else "—")
                L.append(f"| {name} | {lv[1:]}% | {v} | {cell('missed', 'false_alarm_rate')} | {cell('missed', 'recall')} | "
                         f"{cell('over', 'false_alarm_rate')} | {cell('over', 'recall')} |")
    return "\n".join(L) + "\n"


def plot(out: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"p2_full": "#2a78d6", "p2_dropout": "#eb6834"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for ax, (kind, xlabel) in zip(axes, (("erode", "erosion radius (10 m px) → cloud edges missed"),
                                         ("dilate", "dilation radius (10 m px) → clear pixels masked"),
                                         ("miss", "fraction of cloud pixels missed (whole clouds)"))):
        vs = [v for v in VARIANTS if v.startswith(kind)]
        xs = [0] + [int(v[len(kind):]) for v in vs]
        for arm, name in ARMS.items():
            mean = [out["arms"][arm]["clean"]["z30"]["f1"]["mean"]] + [out["arms"][arm]["clean"][f"c30__{v}"]["f1"]["mean"] for v in vs]
            sd = [out["arms"][arm]["clean"]["z30"]["f1"]["sd"]] + [out["arms"][arm]["clean"][f"c30__{v}"]["f1"]["sd"] for v in vs]
            ax.errorbar(range(len(xs)), mean, yerr=sd, color=colors[arm], marker="o", markersize=6, linewidth=2, capsize=3, label=f"{name}, fill under mask")
            ax.axhline(out["arms"][arm]["clean"]["c30"]["f1"]["mean"], color=colors[arm], linestyle=":", linewidth=1.5,
                       label=f"{name}, cloud, no fill")
        ax.set_xticks(range(len(xs)), ["oracle"] + [str(x) + ("%" if kind == "miss" else "") for x in xs[1:]])
        ax.set_xlabel(xlabel, fontsize=8)
        ax.grid(axis="y", color="#e0e0e0", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("test F1 at 30% cloud, frozen clean threshold\n(mean ± SD, 10 seeds)", fontsize=8)
    axes[2].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "curves.png", dpi=150)


if __name__ == "__main__":
    main()
