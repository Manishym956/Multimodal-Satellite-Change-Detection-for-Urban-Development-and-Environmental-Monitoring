"""Build experiments/robust/REPORT.md from REPORT.template.md and the files in experiments/robust/.

Markers: <!-- INCLUDE <file> --> pastes a generated table file; the other markers are
computed here from metrics.json / reeval.json / gap_analysis.json / summary_*.json.
The Phase 2 interpretation applies the pre-registered claim rule mechanically.

Usage:
    .venv/Scripts/python.exe build_robust_report.py
"""

import json
import re
import statistics
from pathlib import Path

import torch
from scipy import stats

ROOT = Path(__file__).resolve().parent
R = ROOT / "experiments" / "robust"
CONDITIONS = ("full", "no_o2", "o2_copy", "no_opt")
PUBLISHED = 0.600


def fmt_p(p: float) -> str:
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def reeval_summary() -> str:
    files = sorted(R.glob("*_seed*/reeval.json"))
    bad = [f.parent.name for f in files
           if not all(v if isinstance(v, bool) else v["exact"] for v in load(f)["reproduction"].values())]
    return (f"{len(files)} runs re-evaluated; {len(files) - len(bad)} reproduced every stored count and threshold exactly"
            + (f"; **mismatch in {bad}**" if bad else "") + ".")


def codecheck() -> str:
    a = torch.load(ROOT / "checkpoints/robust/hafner_full_seed1.pt", map_location="cpu", weights_only=False)["model_state_dict"]
    b = torch.load(ROOT / "checkpoints/robust/codecheck_hafner_full_seed1.pt", map_location="cpu", weights_only=False)["model_state_dict"]
    same_w = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    ma, mb = load(R / "hafner_full_seed1/metrics.json"), load(R / "codecheck_hafner_full_seed1/metrics.json")
    same_m = all(ma["test"][c] == mb["test"][c] for c in ma["test"])
    return f"weights bit-identical: **{same_w}**; stored test metrics identical: **{same_m}**."


def gap_table() -> str:
    g = load(R / "gap_analysis.json")
    lines = ["| seed | F1 @ train-city threshold (τ) | F1 @ 0.66 | F1 @ 0.5 |", "|---|---|---|---|"]
    for r in sorted(g["runs"], key=lambda r: r["seed"]):
        lines.append(f"| {r['seed']} | {r['f1_at_train_threshold']:.3f} ({r['train_threshold']:.2f}) | {r['f1_at_0.66']:.3f} | {r['f1_at_0.5']:.3f} |")
    lines += ["", f"| protocol label | n | mean ± SD | min–max | seeds ≥ {PUBLISHED:.3f} | one-sample t vs {PUBLISHED:.3f} (p) |",
              "|---|---|---|---|---|---|"]
    for key, label in (("f1_at_train_threshold", "final step, train-city threshold (train_network.py)"),
                       ("f1_at_0.66", "final step, fixed 0.66 (evaluation.py THRESH)"), ("f1_at_0.5", "final step, fixed 0.5")):
        vals = [r[key] for r in g["runs"]]
        a = g["aggregate"][key]
        lines.append(f"| {label} | {a['n']} | {a['mean']:.3f} ± {a['sd']:.3f} | {a['min']:.3f}–{a['max']:.3f} | "
                     f"{sum(v >= PUBLISHED for v in vals)} | {stats.ttest_1samp(vals, PUBLISHED).pvalue:.4f} |")
    return "\n".join(lines)


def step_val() -> str:
    out = []
    for name, steps in (("holdout_full", 390), ("holdout_full_s1000", 1000), ("holdout_full_s2500", 2500)):
        v = [load(f)["selection_score"] for f in sorted(R.glob(f"{name}_seed*/metrics.json"))]
        out.append(f"{steps} steps {statistics.mean(v):.3f} (n={len(v)})")
    return "; ".join(out)


def p2_tables(summary: dict) -> str:
    if not {"p2_full", "p2_dropout"} <= summary.keys():
        return "_Phase 2 not finished._"
    a, b = summary["p2_full"], summary["p2_dropout"]
    lines = []
    for at, label in (("at_selected", "the validation-selected threshold"), ("at_0.5", "threshold 0.5")):
        lines += [f"Per-seed test F1 at {label} (A = p2_full, B = p2_dropout):", "",
                  "| seed | " + " | ".join(f"{c} A | {c} B" for c in CONDITIONS) + " |", "|---|" + "---|---|" * len(CONDITIONS)]
        for s in a["seeds"]:
            cells = []
            for c in CONDITIONS:
                for arm in (a, b):
                    e = arm["conditions"][c]
                    cells.append(f"{e[at]['f1']['values'][e['seeds'].index(s)]:.3f}" if s in e["seeds"] else "—")
            lines.append(f"| {s} | " + " | ".join(cells) + " |")
        lines += ["", f"Precision / recall / IoU at {label}, mean ± sample SD over 10 seeds:", "",
                  "| arm | condition | F1 | precision | recall | IoU |" + (" mean τ |" if at == "at_selected" else ""),
                  "|---|---|---|---|---|---|" + ("---|" if at == "at_selected" else "")]
        for name, arm in (("A p2_full", a), ("B p2_dropout", b)):
            for c in CONDITIONS:
                e = arm["conditions"][c]
                cell = lambda m: f"{e[at][m]['mean']:.3f} ± {e[at][m]['sd']:.3f}"
                lines.append(f"| {name} | {c} | {cell('f1')} | {cell('precision')} | {cell('recall')} | {cell('iou')} |"
                             + (f" {statistics.mean(e['selected_threshold']):.2f} |" if at == "at_selected" else ""))
        lines.append("")
    lines += ["Per-seed selected checkpoint step: A " + ", ".join(map(str, a["selected_steps"]))
              + "; B " + ", ".join(map(str, b["selected_steps"])) + ".",
              "Per-seed confusion counts (TP/FP/FN/TN) for every condition and threshold: `summary_prefix-p2__ref-p2_full.json`."]
    return "\n".join(lines)


def p2_paired() -> str:
    path = R / "p2_audit.json"
    if not path.exists():
        return "_audit not run_"
    au = load(path)
    lines = []
    for at, label in (("at_selected", "validation-selected threshold (**primary, pre-registered**)"),
                      ("at_0.5", "threshold 0.5 (secondary, not pre-registered for inference; descriptive)")):
        lines += [f"Paired B − A, test F1 at {label}, seeds 1–10:", "",
                  "| condition | mean Δ | SD of Δ | 95% CI of Δ | d_z | B > A | paired t p | Holm p (t) | Wilcoxon p | Holm p (Wilcoxon) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for c in CONDITIONS:
            r = au["paired"][at][c]
            lines.append(f"| {c} | {r['mean_diff']:+.3f} | {r['sd_diff']:.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['dz']:+.2f} | "
                         f"{r['b_gt_a']}/10 | {fmt_p(r['t_p'])} | {fmt_p(r['holm_p'])} | {fmt_p(r['wilcoxon_p'])} | {fmt_p(r['holm_wilcoxon_p'])} |")
        lines.append("")
    return "\n".join(lines)


def audit_summary() -> str:
    path = R / "p2_audit.json"
    if not path.exists():
        return "_audit not run_"
    au = load(path)
    return (f"`audit_p2.py`: {au['n_checks']} checks, **{au['n_failed']} failed**"
            + ("" if not au["failed"] else ": " + "; ".join(f["check"] for f in au["failed"][:10])) + ".")


def p2_interpretation(summary: dict) -> str:
    rows = summary.get("p2_dropout", {}).get("paired_vs_p2_full")
    if not rows:
        return "_pending_"
    out = []
    for r in rows:
        if r["holm_p"] < 0.05 and r["mean_diff"] > 0:
            verdict = "B better (pre-registered rule met)"
        elif r["holm_p"] < 0.05:
            verdict = "B worse (significant)"
        else:
            verdict = "no supported difference"
        out.append(f"`{r['condition']}`: Δ(B−A) = {r['mean_diff']:+.3f}, Holm p = {fmt_p(r['holm_p'])}, "
                   f"B > A in {sum(d > 0 for d in r['diffs'])}/{r['n']} seeds → **{verdict}**")
    return "\n\n" + "\n".join(f"- {o}" for o in out)


def main() -> None:
    text = (R / "REPORT.template.md").read_text(encoding="utf-8")
    text = re.sub(r"<!-- INCLUDE (\S+) -->",
                  lambda m: (R / m.group(1)).read_text(encoding="utf-8").strip() if (R / m.group(1)).exists() else f"_missing {m.group(1)}_", text)
    p2_path = R / "summary_prefix-p2__ref-p2_full.json"
    p2 = load(p2_path) if p2_path.exists() else {}
    env = load(next(R.glob("hafner_full_seed1/metrics.json")))
    fills = {"REEVAL_SUMMARY": reeval_summary(), "CODECHECK": codecheck(), "GAP_TABLE": gap_table(), "STEP_VAL": step_val(),
             "P2_SEEDS": p2_tables(p2), "P2_PAIRED": p2_paired(), "P2_AUDIT": audit_summary(), "P2_INTERPRETATION": p2_interpretation(p2),
             "ENV": f"torch {env['torch']}, GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}, cudnn deterministic."}
    for key, value in fills.items():
        text = text.replace(f"<!-- {key} -->", value)
    assert "<!--" not in text, "unfilled marker"
    (R / "REPORT.md").write_text(text, encoding="utf-8")
    print("wrote experiments/robust/REPORT.md")


if __name__ == "__main__":
    main()
