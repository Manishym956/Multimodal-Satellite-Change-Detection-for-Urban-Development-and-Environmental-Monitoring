"""Update fig01 to show the clean async CNN with error bar (3 seeds)."""
import warnings
warnings.filterwarnings("ignore")
import sys, numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures" / "paper"

MODELS = [
    ("SAR-only U-Net",        0.3558, None),
    ("Optical-only U-Net",    0.2663, None),
    ("Early Fusion",          0.3520, None),
    ("Sync Dual-Stream CNN",  0.3868, None),
    ("Naive Async CNN",       0.2052, None),
    ("Clean Async CNN\n(mean ± std)", 0.3215, 0.1240),
    ("Transformer Self-Attn", 0.2357, None),
    ("Transformer Cross-Attn",0.2477, None),
]

COLORS = [
    "#6baed6", "#74c476", "#fd8d3c", "#9ecae1",
    "#fdd49e", "#e31a1c", "#d9d9d9", "#bdbdbd",
]

labels = [m[0] for m in MODELS]
means  = [m[1] for m in MODELS]
stds   = [m[2] for m in MODELS]

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(labels))

bars = ax.bar(x, means, color=COLORS, edgecolor="white", linewidth=0.8, zorder=3)

# Error bar only for clean async CNN (index 5)
for i, (m, s) in enumerate(zip(means, stds)):
    if s is not None:
        ax.errorbar(x[i], m, yerr=s, fmt="none", color="#333333",
                    capsize=6, capthick=2, elinewidth=2, zorder=5)
        # Annotate each seed
        ax.text(x[i] + 0.35, 0.4117 + 0.005, "s42: 0.4117", fontsize=6.5, color="#555")
        ax.text(x[i] + 0.35, 0.3727 + 0.005, "s43: 0.3727", fontsize=6.5, color="#555")
        ax.text(x[i] + 0.35, 0.1801 + 0.005, "s44: 0.1801", fontsize=6.5, color="#c0392b")

for i, (bar, v, s) in enumerate(zip(bars, means, stds)):
    label = f"{v:.3f}" if s is None else f"{v:.3f}"
    ypos = v + (s if s else 0) + 0.006
    ax.text(bar.get_x() + bar.get_width() / 2, ypos, label,
            ha="center", va="bottom", fontsize=8,
            fontweight="bold" if i == 5 else "normal",
            color="#e31a1c" if i == 5 else "black")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
ax.set_ylabel("F1 Score (τ = 0.5)", fontsize=11)
ax.set_title(
    "OSCD Test F1 — All Model Configurations\n"
    "Clean Async CNN shown as mean ± std over 3 seeds (42/43/44); all others seed 42",
    fontsize=11
)
ax.set_ylim(0, 0.62)
ax.axvline(5.5, color="#999999", linestyle=":", linewidth=1, alpha=0.5)
ax.annotate("pos_weight=1 (CNNs)", xy=(2.5, 0.01), fontsize=7.5, color="#888",
            ha="center", style="italic")
ax.annotate("pos_weight=41 (Transformers)", xy=(6.5, 0.01), fontsize=7.5, color="#888",
            ha="center", style="italic")
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out = OUT / "fig01_f1_bar.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Updated: figures/paper/fig01_f1_bar.png")
