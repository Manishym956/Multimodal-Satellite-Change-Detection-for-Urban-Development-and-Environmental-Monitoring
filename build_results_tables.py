"""Build the paper's OSCD result tables and seed statistics from saved metrics.

Reads experiments/<name>_seed*/metrics.json (via the multiseed summaries) and the
Transformer tuning CSV, then rewrites the blocks between the markers
<!-- AUTO:<KEY>:BEGIN --> ... <!-- AUTO:<KEY>:END --> in paper/RESEARCH_PAPER_FINAL.md.
Also writes experiments/paper_statistics.json.

Usage:
    .venv/Scripts/python.exe build_results_tables.py
"""

import csv
import json
import re
import statistics
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments"
PAPER = ROOT / "paper" / "RESEARCH_PAPER_FINAL.md"
KEYS = ["final_test_f1", "final_test_precision", "final_test_recall", "final_test_iou", "final_test_far"]

# (table label, experiment name, params, loss note)
MODELS = [
    ("SAR-only U-Net", "sar_baseline", "31.04M", "pw=1"),
    ("Optical-only U-Net", "optical_baseline", "31.05M", "pw=1"),
    ("Early Fusion U-Net", "fusion_baseline", "31.05M", "pw=1"),
    ("Sync Dual-Stream CNN", "dual_stream_baseline", "27.28M", "pw=1"),
    ("Naive Missing-Optical CNN", "asynchronous_dual_stream", "27.28M", "pw=1"),
    ("Clean Missing-Optical CNN", "asynchronous_dual_stream_clean", "27.28M", "pw=1"),
    ("Transformer Self-Attn (original)", "transformer_self_attention", "1.36M", "pw=41"),
    ("Transformer Cross-Attn (original)", "transformer_cross_attention", "1.16M", "pw=41"),
    ("Transformer Cross-Attn (pw=1)", "transformer_cross_attention_pw1", "1.16M", "pw=1"),
]


def per_seed(name: str) -> dict[int, dict]:
    out = {}
    for m in sorted(EXP.glob(f"{name}_seed*/metrics.json")):
        doc = json.loads(m.read_text(encoding="utf-8"))
        out[int(doc["reproducibility"]["seed"])] = doc
    return out


def tuned_transformers() -> dict[str, dict[int, dict]]:
    """Tuned (weight decay 1e-4) Transformer runs at the common threshold 0.5."""
    out: dict[str, dict[int, dict]] = {"self": {}, "cross": {}}
    path = EXP / "transformer_tuning_2026-10-15" / "final_test_comparison.csv"
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["dataset"] == "oscd" and r["checkpoint"] == "tuned" and float(r["threshold"]) == 0.5:
                out[r["architecture"]][int(r["seed"])] = {
                    "final_test_f1": float(r["f1"]), "final_test_precision": float(r["precision"]),
                    "final_test_recall": float(r["recall"]), "final_test_iou": float(r["iou"]),
                    "final_test_far": float(r["false_alarm_rate"]), "validation_best_epoch": None,
                }
    return out


def agg(runs: dict[int, dict]) -> dict:
    res = {"seeds": sorted(runs), "n": len(runs)}
    for k in KEYS:
        vals = [runs[s][k] for s in sorted(runs)]
        res[k] = vals
        res[k + "_mean"] = statistics.mean(vals)
        res[k + "_sd"] = statistics.stdev(vals) if len(vals) > 1 else None
    res["best_epochs"] = [runs[s].get("validation_best_epoch") for s in sorted(runs)]
    return res


def cell(a: dict, k: str, d: int = 3) -> str:
    m, s = a[k + "_mean"], a[k + "_sd"]
    return f"{m:.{d}f}" if s is None else f"{m:.{d}f} ± {s:.{d}f}"


def welch(a: list[float], b: list[float]) -> dict:
    t = stats.ttest_ind(a, b, equal_var=False)
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"mean_diff": statistics.mean(a) - statistics.mean(b), "welch_t": float(t.statistic),
            "welch_p": float(t.pvalue), "mannwhitney_U": float(u.statistic), "mannwhitney_p": float(u.pvalue),
            "n_a": len(a), "n_b": len(b)}


def main() -> None:
    results = {name: agg(per_seed(name)) for _, name, _, _ in MODELS}
    tuned = {k: agg(v) for k, v in tuned_transformers().items()}

    rows = ["| Model | Loss | n | F1 | Precision | Recall | IoU | FAR | Params | Per-seed F1 |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for label, name, params, loss in MODELS:
        a = results[name]
        seeds = ", ".join(f"{v:.3f}" for v in a["final_test_f1"])
        rows.append(f"| {label} | {loss} | {a['n']} | {cell(a, 'final_test_f1')} | {cell(a, 'final_test_precision')} | "
                    f"{cell(a, 'final_test_recall')} | {cell(a, 'final_test_iou')} | {cell(a, 'final_test_far')} | {params} | {seeds} |")
    for arch, label, params in (("self", "Transformer Self-Attn (tuned, wd=1e-4)", "1.36M"),
                                ("cross", "Transformer Cross-Attn (tuned, wd=1e-4)", "1.16M")):
        a = tuned[arch]
        seeds = ", ".join(f"{v:.3f}" for v in a["final_test_f1"])
        rows.append(f"| {label} | pw=41 | {a['n']} | {cell(a, 'final_test_f1')} | {cell(a, 'final_test_precision')} | "
                    f"{cell(a, 'final_test_recall')} | {cell(a, 'final_test_iou')} | {cell(a, 'final_test_far')} | {params} | {seeds} |")
    table1 = "\n".join(rows)

    f1 = {name: results[name]["final_test_f1"] for _, name, _, _ in MODELS}
    tests = {
        "sync_vs_clean_async": welch(f1["dual_stream_baseline"], f1["asynchronous_dual_stream_clean"]),
        "clean_async_vs_sar": welch(f1["asynchronous_dual_stream_clean"], f1["sar_baseline"]),
        "sync_vs_sar": welch(f1["dual_stream_baseline"], f1["sar_baseline"]),
        "sync_vs_early_fusion": welch(f1["dual_stream_baseline"], f1["fusion_baseline"]),
        "clean_vs_naive_async": welch(f1["asynchronous_dual_stream_clean"], f1["asynchronous_dual_stream"]),
        "xattn_pw41_vs_pw1": welch(f1["transformer_cross_attention"], f1["transformer_cross_attention_pw1"]),
    }
    # Holm step-down adjustment of the Welch p-values across the family of comparisons
    order = sorted(tests, key=lambda k: tests[k]["welch_p"])
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * tests[k]["welch_p"]))
        tests[k]["welch_p_holm"] = running
    trows = ["| Comparison (F1) | Δ mean | n | Welch t | p (Welch) | p (Holm) | p (Mann–Whitney) |", "|---|---|---|---|---|---|---|"]
    names = {"sync_vs_clean_async": "Sync dual-stream − Clean missing-optical",
             "clean_async_vs_sar": "Clean missing-optical − SAR-only",
             "sync_vs_sar": "Sync dual-stream − SAR-only",
             "sync_vs_early_fusion": "Sync dual-stream − Early fusion",
             "clean_vs_naive_async": "Clean − Naive missing-optical",
             "xattn_pw41_vs_pw1": "Transformer Cross-Attn pw=41 − pw=1"}
    for k, t in tests.items():
        trows.append(f"| {names[k]} | {t['mean_diff']:+.3f} | {t['n_a']} vs {t['n_b']} | {t['welch_t']:.2f} | "
                     f"{t['welch_p']:.3f} | {t['welch_p_holm']:.3f} | {t['mannwhitney_p']:.3f} |")
    table_tests = "\n".join(trows)

    (EXP / "paper_statistics.json").write_text(
        json.dumps({"models": results, "tuned_transformers": tuned, "tests": tests}, indent=2), encoding="utf-8")

    text = PAPER.read_text(encoding="utf-8")
    for key, block in (("TABLE1", table1), ("TESTS", table_tests)):
        pattern = re.compile(rf"(<!-- AUTO:{key}:BEGIN -->\n).*?(<!-- AUTO:{key}:END -->)", re.S)
        if not pattern.search(text):
            raise SystemExit(f"marker AUTO:{key} not found in paper")
        text = pattern.sub(lambda m: m.group(1) + block + "\n" + m.group(2), text)
    PAPER.write_text(text, encoding="utf-8")
    print(table1, "\n\n", table_tests, sep="")


if __name__ == "__main__":
    main()
