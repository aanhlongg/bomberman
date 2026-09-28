import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path("results/tree_agent/task2")
OUT_DIR = RESULTS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

ACCENT = "#1f4e79"
WARN = "#c0392b"
GOOD = "#2e8b57"
GREY = "#8a8a8a"


def coins_per_round(eval_json_path: Path) -> float:
    d = json.load(open(eval_json_path))
    stats = next(iter(d["by_agent"].values()))
    return stats["coins"] / stats["rounds"]


configs = [
    ("Pre-bonus\nbaseline", coins_per_round(RESULTS_DIR / "task2_final_eval_v4_routing.json"), ACCENT),
    ("Bonus=1.5", coins_per_round(RESULTS_DIR / "task2_final_eval_v6_safetymask.json"), WARN),
    ("Bonus=0.5", coins_per_round(RESULTS_DIR / "task2_final_eval_v7_extracratebonus05.json"), WARN),
    ("Bonus=0.0\n(final)", coins_per_round(RESULTS_DIR / "task2_final_eval_v8_ablation_zero.json"), GOOD),
]

fig, ax = plt.subplots(figsize=(7.5, 4.5))
labels = [c[0] for c in configs]
values = [c[1] for c in configs]
colors = [c[2] for c in configs]
bars = ax.bar(labels, values, color=colors, width=0.6)
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=10)

ax.set_ylabel("Coins/round (200-round pure-greedy eval)")
ax.set_title("The EXTRA_CRATE_BONUS ablation: isolating the regression's cause")
ax.axhline(50, color=GREY, lw=1, ls=":")
ax.text(len(configs) - 0.4, 50.5, "max = 50", color=GREY, fontsize=9, ha="right")
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig(OUT_DIR / "config_comparison_bar.png", bbox_inches="tight")
plt.close(fig)
print("saved config_comparison_bar.png")
