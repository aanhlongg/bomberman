"""
plot_routing.py -- figures for the tree_agent Task 2 routing revision
=====================================================================
Run from the repo root:

    uv run python metrics/tree_agent/task2/plot_routing.py

Reads only already-saved data under results/tree_agent/task2/ (nothing is
retrained or re-traced here):

  routing_trace_previous_model.json / routing_trace_shipped_model.json
      -- trace_step_budget.py --json output, 100 greedy rounds each
  routing_screen_<variant>_train.json / _eval.json
      -- --save-stats of the 1,500-round screening trainings and their
         50-round pure-greedy evaluations (A control, B, C, D/D2/D3, E, G)
  routing_final_eval200.json          -- shipped model, 200 greedy rounds
  task2_final_eval_v8_ablation_zero.json -- previous model, 200 greedy rounds

and renders four figures next to them:

  routing_step_budget.png       where the 400 steps of a round go, previous vs shipped
  routing_screening_bar.png     coins/round + full clears per screening variant
  routing_learning_curves.png   training-time coins/round per variant (rolling mean)
  routing_eval_distribution.png per-round coins and round length, previous vs shipped
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results/tree_agent/task2")
OUT_DIR = RESULTS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Categorical palette in fixed slot order (blue, orange, aqua, yellow, magenta,
# green) plus ink/surface tokens; identity is never carried by color alone --
# every figure also has a legend and/or direct labels.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "text.color": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})


def load(name):
    return json.load(open(RESULTS_DIR / name))


def eval_rounds(name, agent_key=None):
    d = load(name)
    agent = d["by_agent"][agent_key] if agent_key else next(iter(d["by_agent"].values()))
    rounds = list(d["by_round"].values())
    return agent, rounds


# ---------------------------------------------------------------------------
# 1. step budget: previous vs shipped
# ---------------------------------------------------------------------------
prev = load("routing_trace_previous_model.json")
ship = load("routing_trace_shipped_model.json")
classes = [
    ("coin_chase", "walking to a coin", BLUE),
    ("crate_walk", "walking to a bomb spot", AQUA),
    ("escape", "escaping own bomb", ORANGE),
    ("bomb", "BOMB", YELLOW),
    ("wait", "WAIT", MAGENTA),
]

fig, ax = plt.subplots(figsize=(8.2, 3.3))
rows = [("previous model\n(44.6 coins/round)", prev), ("shipped model\n(50.0 coins/round)", ship)]
for y, (label, trace) in enumerate(rows):
    left = 0.0
    budget = trace["steps_per_round_by_class"]
    other = sum(v for k, v in budget.items() if k not in dict((c[0], 1) for c in classes))
    for key, name, color in classes + [("other", "other", MUTED)]:
        width = other if key == "other" else budget[key]
        ax.barh(y, width, left=left, height=0.52, color=color, edgecolor=SURFACE, linewidth=2)
        if width >= 22:
            ax.text(left + width / 2, y, f"{width:.0f}", ha="center", va="center", color="white", fontsize=9)
        left += width
    ax.text(left + 4, y, f"{left:.0f} steps/round\n{trace['steps_per_bomb_cycle_mean']:.1f} steps per bomb cycle",
            va="center", color=INK_2, fontsize=9)
ax.set_yticks([0, 1])
ax.set_yticklabels([r[0] for r in rows])
ax.set_xlim(0, 520)
ax.set_xlabel("steps per round (mean over 100 pure-greedy loot-crate rounds, seed 0)")
ax.grid(axis="y", visible=False)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c[2]) for c in classes],
          labels=[c[1] for c in classes], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False)
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(OUT_DIR / "routing_step_budget.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 2. screening: coins/round and full clears per variant (one axis)
# ---------------------------------------------------------------------------
variants = [
    ("A", "previous code\n(control)", MUTED),
    ("B", "(1)+(2)\ntargeting +\nfeatures", INK_2),
    ("C", "(1)-(4)\n+ antisym. bonus\n+ 100k replay", INK_2),
    ("D", "(1)-(5)\nseed 1", BLUE),
    ("D2", "(1)-(5)\nseed 2", BLUE),
    ("D3", "(1)-(5)\nseed 3", BLUE),
    ("E", "(1)-(5)\n$\\gamma$ = 0.90", INK_2),
    ("G", "(1)-(5)\ndist. weight 1.0", INK_2),
]
fig, ax = plt.subplots(figsize=(8.6, 4.0))
xs = np.arange(len(variants))
for x, (key, label, color) in zip(xs, variants):
    agent, rounds = eval_rounds(f"routing_screen_{key}_eval.json")
    n = agent["rounds"]
    coins = agent.get("coins", 0) / n
    clears = sum(r["coins"] >= 50 for r in rounds)
    ax.bar(x, coins, width=0.62, color=color, edgecolor=SURFACE, linewidth=2)
    ax.text(x, coins + 1.0, f"{coins:.1f}", ha="center", va="bottom", color=INK, fontsize=9)
    ax.text(x, 1.2 if coins > 6 else coins + 4.5, f"{clears}/{n}\nclears", ha="center", va="bottom",
            color="white" if coins > 6 else INK_2, fontsize=8)
ax.set_xticks(xs)
ax.set_xticklabels([v[1] for v in variants], fontsize=8.5)
ax.set_ylim(0, 56)
ax.axhline(50, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
ax.text(len(variants) - 0.5, 50.6, "all 50 coins", ha="right", va="bottom", color=INK_2, fontsize=8.5)
ax.set_ylabel("coins per round (50-round pure-greedy eval)")
ax.set_title("Screening of the routing revision, 1,500 training rounds each on loot-crate", loc="left")
ax.grid(axis="x", visible=False)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=INK_2),
                   plt.Rectangle((0, 0), 1, 1, color=MUTED)],
          labels=["shipped configuration", "intermediate / knob variant", "previous code"],
          loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(OUT_DIR / "routing_screening_bar.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 3. training-time learning curves (rolling mean of coins/round)
# ---------------------------------------------------------------------------
def training_curve(key, window=50):
    d = load(f"routing_screen_{key}_train.json")
    coins = np.array([r["coins"] for r in d["by_round"].values()], dtype=float)
    kernel = np.ones(window) / window
    smoothed = np.convolve(coins, kernel, mode="valid")
    return np.arange(window, len(coins) + 1), smoothed

series = [
    ("A", "previous code (control)", MUTED, 1.6),
    ("B", "(1)+(2) targeting + features", AQUA, 1.6),
    ("C", "(1)-(4)", ORANGE, 1.6),
    ("D", "(1)-(5) shipped, 3 seeds", BLUE, 1.6),
    ("D2", None, BLUE, 1.0),
    ("D3", None, BLUE, 1.0),
]
fig, ax = plt.subplots(figsize=(8.2, 4.0))
for key, label, color, lw in series:
    x, y = training_curve(key)
    ax.plot(x, y, color=color, linewidth=lw, label=label, alpha=1.0 if label else 0.6)
    if label:
        ax.text(x[-1] + 12, y[-1], label.split(" (")[0] if key != "D" else "(1)-(5) shipped", va="center", color=color, fontsize=8.5)
ax.set_xlim(0, 1780)
ax.set_ylim(0, 52)
ax.set_xlabel("training round")
ax.set_ylabel("coins per round, 50-round rolling mean (training, $\\epsilon$-greedy)")
ax.set_title("Training-time coins/round -- note the control looks alive here and scores 0 at pure-greedy eval", loc="left", fontsize=10.5)
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(OUT_DIR / "routing_learning_curves.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 4. per-round distributions: previous vs shipped model (200 greedy rounds each)
# ---------------------------------------------------------------------------
prev_agent, prev_rounds = eval_rounds("task2_final_eval_v8_ablation_zero.json")
ship_agent, ship_rounds = eval_rounds("routing_final_eval200.json")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
bins = np.arange(29.5, 51.5, 1.0)
ax1.hist([r["coins"] for r in prev_rounds], bins=bins, color=ORANGE, alpha=0.85, label="previous model", edgecolor=SURFACE, linewidth=1)
ax1.hist([r["coins"] for r in ship_rounds], bins=bins, color=BLUE, alpha=0.85, label="shipped model", edgecolor=SURFACE, linewidth=1)
ax1.set_xlabel("coins collected in the round")
ax1.set_ylabel("rounds (of 200)")
ax1.set_title("Coins per round", loc="left")
ax1.legend(loc="upper left", frameon=False)
sbins = np.arange(290, 411, 10)
ax2.hist([r["steps"] for r in prev_rounds], bins=sbins, color=ORANGE, alpha=0.85, label="previous model", edgecolor=SURFACE, linewidth=1)
ax2.hist([r["steps"] for r in ship_rounds], bins=sbins, color=BLUE, alpha=0.85, label="shipped model", edgecolor=SURFACE, linewidth=1)
ax2.set_xlabel("round length in steps (a round ends when the board is clear)")
ax2.set_title("Round length", loc="left")
ax2.legend(loc="upper left", frameon=False)
for ax in (ax1, ax2):
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 215)
fig.tight_layout()
fig.savefig(OUT_DIR / "routing_eval_distribution.png", bbox_inches="tight")
plt.close(fig)

print("wrote", ", ".join(p.name for p in sorted(OUT_DIR.glob("routing_*.png"))))
