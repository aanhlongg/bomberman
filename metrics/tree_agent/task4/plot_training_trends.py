"""
plot_training_trends.py -- training-time trends of tree_agent runs
====================================================================
Run from the repo root, e.g. while or after training:

    uv run python metrics/tree_agent/task4/plot_training_trends.py \
        --run "seed 1=agent_code/tree_ab_L1" --run "seed 2=agent_code/tree_ab_L2" \
        --out results/tree_agent/task4/training_trends.png

Reads, per run, the agent's own per-step metrics/training_log.csv (our score
delta per step: +1 coin, +5 kill) and tracked_bomb_log.jsonl (own-bomb
deaths), and renders three panels over training rounds: our score per round,
own-bomb death rate, and round length -- each as a rolling mean over WINDOW
rounds.

Read with care: these are TRAINING-TIME numbers under epsilon-greedy
exploration (epsilon floor 0.05 from round ~1,000). A decline here is real
(a policy collapse shows up as the score sinking to what exploration alone
earns); a plateau here does NOT certify a greedy plateau -- for that, evaluate
checkpoints pure-greedy (select_checkpoint.py).
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1", "#fcfcfb"
COLORS = [BLUE, ORANGE, AQUA, YELLOW]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 9, "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
})


def load_run(agent_dir: Path):
    log = pd.read_csv(agent_dir / "metrics" / "training_log.csv", usecols=["round", "step", "sparse_reward"])
    per_round = log.groupby("round").agg(score=("sparse_reward", "sum"), steps=("step", "max"))
    deaths = pd.Series(0, index=per_round.index, dtype=float)
    bomb_log = agent_dir / "tracked_bomb_log.jsonl"
    if bomb_log.exists():
        for line in open(bomb_log):
            rec = json.loads(line)
            if not rec.get("survived", True) and rec["round"] in deaths.index:
                deaths[rec["round"]] = 1.0
    per_round["own_death"] = deaths
    # the final round may be partial while training is still running
    return per_round.iloc[:-1] if len(per_round) > 1 else per_round


def rolling(series, window):
    return series.rolling(window, min_periods=max(10, window // 5)).mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help='"label=agent_dir" (repeatable)')
    ap.add_argument("--window", type=int, default=250)
    ap.add_argument("--checkpoint-interval", type=int, default=250, help="draw faint ticks at checkpoint rounds (0: none)")
    ap.add_argument("--out", default="results/tree_agent/task4/training_trends.png")
    ap.add_argument("--title", default="tree_agent vs rule_based_agent -- training-time trends ($\\epsilon$-greedy, rolling mean)")
    a = ap.parse_args()

    runs = []
    for spec in a.run:
        label, path = spec.split("=", 1)
        runs.append((label, load_run(Path(path))))

    fig, axes = plt.subplots(3, 1, figsize=(8.6, 8.4), sharex=True)
    ax_score, ax_death, ax_len = axes
    max_round = 0
    for (label, df), color in zip(runs, COLORS):
        max_round = max(max_round, int(df.index.max()))
        for ax, col in ((ax_score, "score"), (ax_death, "own_death"), (ax_len, "steps")):
            y = rolling(df[col], a.window)
            ax.plot(df.index, y, color=color, linewidth=1.6, label=f"{label} ({int(df.index.max()):,} rounds)")
            last = y.dropna()
            if len(last):
                ax.text(last.index[-1] + max_round * 0.01, last.iloc[-1], label, color=color, va="center", fontsize=8.5)
    ax_score.set_ylabel("our score per round\n(coins + 5 per kill)")
    ax_death.set_ylabel("own-bomb death rate")
    ax_death.set_ylim(0, max(0.05, ax_death.get_ylim()[1]))
    ax_len.set_ylabel("round length (steps)")
    ax_len.set_xlabel("training round")
    ax_score.set_title(a.title, loc="left")
    for ax in axes:
        ax.axvline(1000, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
        if a.checkpoint_interval:
            for r in range(a.checkpoint_interval, max_round + 1, a.checkpoint_interval):
                ax.axvline(r, color=GRID, linewidth=0.5, zorder=0)
        ax.grid(axis="x", visible=False)
    ax_score.text(1000, ax_score.get_ylim()[1], " $\\epsilon$ reaches its 0.05 floor", color=INK_2, fontsize=8.5, va="top")
    ax_score.legend(loc="lower right", frameon=False)
    ax_score.set_xlim(0, max_round * 1.08)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print("wrote", a.out)
    for label, df in runs:
        late = df.iloc[-a.window:]
        print(f"{label}: rounds {int(df.index.max()):,}; last {len(late)} rounds: score/round {late['score'].mean():.2f}, own-bomb deaths {late['own_death'].mean():.1%}, steps/round {late['steps'].mean():.0f}")


if __name__ == "__main__":
    main()
