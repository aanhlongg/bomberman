"""
plot_report_extras.py -- additional evaluation figures for the shortened report
==================================================================================
Run from the repo root:

    python plot_report_extras.py

Produces:
  results/linear_agent/task1/task1_final_eval_distribution.png
      -- Task 1 (coin-heaven), final-100-training-round coin distribution
         for the sweep-winning config (alpha=0.05, gamma=0.95), derived from
         the archived sweep log (results/linear_agent/task1/log_a0.05_g0.95.csv).
  results/linear_agent/task2/final_agent_eval_distribution.png
      -- Task 2 (loot-crate), 200-round pure-greedy coin distribution for the
         final validated agent (escape-routing fix + original 50k weights).
  results/linear_agent/task2/steps_used_before_after.png
      -- Task 2, steps used per round: before the escape-routing fix (always
         exactly 400 -- no early finishes) vs after (perfect-clear rounds
         finish early, some in as few as 356 steps).
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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


# ── Task 1: final-100-round coin distribution (sweep-winning config) ───────

coins_per_round = defaultdict(float)
with open("results/linear_agent/task1/log_a0.05_g0.95.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        coins_per_round[int(row["round"])] += float(row["sparse_reward"])

rounds = sorted(coins_per_round.keys())
last100 = np.array([coins_per_round[r] for r in rounds[-100:]])

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.hist(last100, bins=np.arange(20, 52, 2), color=ACCENT, alpha=0.8)
ax.axvline(last100.mean(), color=WARN, ls="--", lw=1.5, label=f"mean = {last100.mean():.1f}")
ax.set_xlabel("Coins collected per round (final 100 training rounds)")
ax.set_ylabel("Number of rounds")
ax.set_title(r"Task 1: converged performance ($\alpha$=0.05, $\gamma$=0.95)")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
Path("results/linear_agent/task1").mkdir(parents=True, exist_ok=True)
fig.savefig("results/linear_agent/task1/task1_final_eval_distribution.png", bbox_inches="tight")
plt.close(fig)
print("saved results/linear_agent/task1/task1_final_eval_distribution.png")


# ── Task 2: final agent eval distribution ───────────────────────────────────

with open("results/escape_routing_test_400steps.json") as f:
    d = json.load(f)
coins = np.array([v["coins"] for v in d["by_round"].values()])

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.hist(coins, bins=np.arange(20, 52, 2), color=GOOD, alpha=0.8)
ax.axvline(coins.mean(), color=WARN, ls="--", lw=1.5, label=f"mean = {coins.mean():.1f}")
ax.set_xlabel("Coins collected per round (200-round pure-greedy eval)")
ax.set_ylabel("Number of rounds")
ax.set_title("Task 2: final validated agent")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
Path("results/linear_agent/task2").mkdir(parents=True, exist_ok=True)
fig.savefig("results/linear_agent/task2/final_agent_eval_distribution.png", bbox_inches="tight")
plt.close(fig)
print("saved results/linear_agent/task2/final_agent_eval_distribution.png")


# ── Task 2: steps used before/after the escape-routing fix ─────────────────

with open("results/crate_hit_bonus5_50k_eval.json") as f:
    before = json.load(f)
steps_before = np.array([v["steps"] for v in before["by_round"].values()])

with open("results/escape_routing_test_400steps.json") as f:
    after = json.load(f)
steps_after = np.array([v["steps"] for v in after["by_round"].values()])

def ecdf(values):
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

fig, ax = plt.subplots(figsize=(7, 4.2))
xb, yb = ecdf(steps_before)
xa, ya = ecdf(steps_after)
ax.step(xb, yb, where="post", color=WARN, lw=2.2, label=f"Before fix (mean {steps_before.mean():.0f})")
ax.step(xa, ya, where="post", color=GOOD, lw=2.2, label=f"After fix (mean {steps_after.mean():.0f})")
ax.set_xlabel("Steps used per round (out of 400)")
ax.set_ylabel("Fraction of rounds finished by this step")
ax.set_title("Task 2: escape-routing fix lets rounds finish early")
ax.set_xlim(345, 402)
ax.legend(loc="upper left")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("results/linear_agent/task2/steps_used_before_after.png", bbox_inches="tight")
plt.close(fig)
print("saved results/linear_agent/task2/steps_used_before_after.png")
