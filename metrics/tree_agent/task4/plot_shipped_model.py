"""
plot_shipped_model.py -- the shipped tree_agent model across every matchup
===========================================================================
Run from the repo root:

    uv run python metrics/tree_agent/task4/plot_shipped_model.py

Reads the archived 200-round pure-greedy evaluations under results/tree_agent/task4/
(nothing is re-run) and renders shipped_model_matchups.png: score per round, kill
rate and own-bomb suicide rate of the shipped model (5,000 rounds vs rule_based_agent,
the routing revision + Task 4 escape fixes) against the previous Task 4 model, on
1v1 vs rule_based_agent, the 4-player free-for-all, and the two Task 3 opponents.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results/tree_agent/task4")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 9, "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
})

MATCHUPS = [
    # label, previous-model file, shipped-model file
    ("1v1\nrule_based", "task4_rulebased_eval.json", "task4_v2_Frule_eval.json"),
    ("FFA vs 3x\nrule_based", "task4_ffa.json", "task4_v2_Frule_ffa.json"),
    ("vs\npeaceful", "task4_regression_peaceful.json", "task4_v2_Frule_vs_peace.json"),
    ("vs\ncoin_collector", "task4_regression_collector.json", "task4_v2_Frule_vs_coll.json"),
]


def stats(path):
    d = json.load(open(RESULTS / path))
    me_key = next(k for k in d["by_agent"] if k.startswith("tree"))
    me = d["by_agent"][me_key]; n = me["rounds"]
    opp = [a for k, a in d["by_agent"].items() if k != me_key]
    return {
        "score": me.get("score", 0) / n,
        "opp_score": sum(a.get("score", 0) for a in opp) / len(opp) / n,
        "kills": me.get("kills", 0) / n,
        "suicides": me.get("suicides", 0) / n,
    }


prev = [stats(p) for _, p, _ in MATCHUPS]
ship = [stats(s) for _, _, s in MATCHUPS]
labels = [m[0] for m in MATCHUPS]
x = np.arange(len(MATCHUPS)); w = 0.36

fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.5, 1, 1]})

ax = axes[0]
ax.bar(x - w / 2, [p["score"] for p in prev], w, color=ORANGE, edgecolor=SURFACE, linewidth=2, label="previous Task 4 model")
ax.bar(x + w / 2, [s["score"] for s in ship], w, color=BLUE, edgecolor=SURFACE, linewidth=2, label="shipped model")
ax.scatter(x - w / 2, [p["opp_score"] for p in prev], marker="_", s=260, color=INK, linewidths=1.8, zorder=5, label="opponent's score (avg)")
ax.scatter(x + w / 2, [s["opp_score"] for s in ship], marker="_", s=260, color=INK, linewidths=1.8, zorder=5)
for i in range(len(MATCHUPS)):
    ax.text(x[i] - w / 2, prev[i]["score"] + 0.2, f"{prev[i]['score']:.1f}", ha="center", va="bottom", fontsize=8.5)
    ax.text(x[i] + w / 2, ship[i]["score"] + 0.2, f"{ship[i]['score']:.1f}", ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("score per round (coins + 5 per kill)")
ax.set_title("Score, 200 pure-greedy rounds per matchup", loc="left")
ax.set_ylim(0, 16); ax.legend(frameon=False, loc="upper left"); ax.grid(axis="x", visible=False)

for ax, key, title in ((axes[1], "kills", "Kill rate"), (axes[2], "suicides", "Own-bomb suicide rate")):
    ax.bar(x - w / 2, [p[key] for p in prev], w, color=ORANGE, edgecolor=SURFACE, linewidth=2)
    ax.bar(x + w / 2, [s[key] for s in ship], w, color=BLUE, edgecolor=SURFACE, linewidth=2)
    for i in range(len(MATCHUPS)):
        ax.text(x[i] - w / 2, prev[i][key] + 0.01, f"{prev[i][key]:.0%}", ha="center", va="bottom", fontsize=8)
        ax.text(x[i] + w / 2, ship[i][key] + 0.01, f"{ship[i][key]:.0%}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1.08 if key == "kills" else 0.28)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(title, loc="left"); ax.grid(axis="x", visible=False)

fig.tight_layout()
out = RESULTS / "shipped_model_matchups.png"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
for label, p, s in zip(labels, prev, ship):
    print(f"{label.replace(chr(10), ' '):28} previous {p['score']:5.2f} vs {p['opp_score']:5.2f} (kills {p['kills']:5.1%}, suicides {p['suicides']:5.1%}) | shipped {s['score']:5.2f} vs {s['opp_score']:5.2f} (kills {s['kills']:5.1%}, suicides {s['suicides']:5.1%})")
