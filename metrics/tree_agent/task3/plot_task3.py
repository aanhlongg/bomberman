"""
plot_task3.py -- Task 3 (peaceful_agent / coin_collector_agent) result figures
================================================================================
Run from the repo root:

    uv run metrics/tree_agent/task3/plot_task3.py

Reads exclusively from already-saved results/tree_agent/task3/ eval JSONs and
the archived per-step training log -- does not retrain anything itself.

Produces 2 figures in results/tree_agent/task3/:
  1. opponent_potential_scale_fix.png -- coins/round and bombs/round, broken
                                          (OPPONENT_POTENTIAL_SCALE=1.0) vs.
                                          fixed (0.2), vs. peaceful_agent
  2. learning_curve.png               -- smoothed coins/round and kills/round
                                          over the full 5000-round
                                          coin_collector_agent training run
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "results" / "tree_agent" / "task3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SMOOTH_WIN = 100

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ACCENT = "#1f4e79"
WARN = "#c0392b"
GOOD = "#2e8b57"
GREY = "#8a8a8a"


def load_eval(rel_path: str):
    path = REPO_ROOT / rel_path
    if not path.is_file():
        print(f"  [skip] missing: {rel_path}")
        return None
    with open(path) as f:
        return json.load(f)["by_agent"]["tree_agent"]


# ── Figure 1: OPPONENT_POTENTIAL_SCALE regression, before vs. after ────────

def plot_regression_fix():
    broken = load_eval("results/tree_agent/task3/task3_peaceful_eval.json")
    fixed = load_eval("results/tree_agent/task3/task3_peaceful_eval_v2.json")
    if broken is None or fixed is None:
        return

    configs = [
        ("Scale=1.0\n(unscaled, broken)", broken, WARN),
        ("Scale=0.2\n(fixed)", fixed, GOOD),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    labels = [c[0] for c in configs]
    # the broken (OPPONENT_POTENTIAL_SCALE=1.0) run collected/bombed exactly
    # zero, so those keys are absent from its stats dict entirely (the game
    # engine only includes a key for events that actually occurred) --
    # .get(..., 0) treats "never happened" and "explicitly zero" the same,
    # which is exactly the intended reading here.
    coins_per_round = [c[1].get("coins", 0) / c[1]["rounds"] for c in configs]
    bombs_per_round = [c[1].get("bombs", 0) / c[1]["rounds"] for c in configs]
    colors = [c[2] for c in configs]

    ax = axes[0]
    bars = ax.bar(labels, coins_per_round, color=colors, width=0.6)
    for bar, v in zip(bars, coins_per_round):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.15, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_ylabel("Coins/round (200-round pure-greedy eval)")
    ax.set_title("Coins collected")
    ax.axhline(9, color=GREY, lw=1, ls=":")
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    bars = ax.bar(labels, bombs_per_round, color=colors, width=0.6)
    for bar, v in zip(bars, bombs_per_round):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4, f"{v:.1f}", ha="center", fontsize=10)
    ax.set_ylabel("Bombs/round")
    ax.set_title("Bombs dropped")
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("The OPPONENT_POTENTIAL_SCALE fix, vs. peaceful_agent", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "opponent_potential_scale_fix.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote opponent_potential_scale_fix.png")


# ── Figure 2: learning curve over the coin_collector_agent training run ────

def plot_learning_curve():
    path = OUT_DIR / "training_log_collector.csv"
    if not path.is_file():
        print(f"  [skip] missing: {path}")
        return

    df = pd.read_csv(path)
    # sparse_reward is the per-step SCORE delta (game_state["self"][1] --
    # game score, not the shaped training reward), so it's exactly
    # settings.py's REWARD_COIN=1 on a coin pickup and REWARD_KILL=5 on a
    # kill, 0 otherwise -- unambiguous per-step event markers, not a
    # cumulative or shaped signal.
    coins_per_step = (df["sparse_reward"] == 1).astype(float)
    kills_per_step = (df["sparse_reward"] == 5).astype(float)

    by_round = df.assign(coin=coins_per_step, kill=kills_per_step).groupby("round").agg(
        coins=("coin", "sum"), kills=("kill", "sum")
    )

    def smooth(arr, win):
        if len(arr) < win:
            return arr
        kernel = np.ones(win) / win
        return np.convolve(arr, kernel, mode="valid")

    rounds = by_round.index.to_numpy()
    smoothed_rounds = rounds[SMOOTH_WIN - 1:] if len(rounds) >= SMOOTH_WIN else rounds

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(smoothed_rounds, smooth(by_round["coins"].to_numpy(), SMOOTH_WIN),
             label="coins/round", color="#2196F3")
    ax.plot(smoothed_rounds, smooth(by_round["kills"].to_numpy(), SMOOTH_WIN),
             label="kills/round", color="#4CAF50")
    ax.set_xlabel("Training round")
    ax.set_ylabel(f"Rolling mean (window={SMOOTH_WIN})")
    ax.set_title("Task 3: training curve vs. coin_collector_agent (5000 rounds, OPPONENT_POTENTIAL_SCALE=0.2)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "learning_curve.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote learning_curve.png")


if __name__ == "__main__":
    plot_regression_fix()
    plot_learning_curve()
