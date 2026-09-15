"""
plot_task3.py -- Task 3 (peaceful_agent / coin_collector_agent) result figures
================================================================================
Run from anywhere after cloning (self-locating, resolves the repo root from
its own file path):

    uv run metrics/linear_agent/task3/plot_task3.py

Reads exclusively from already-archived results/archive_task3_*/ eval and
train JSON files (produced by metrics/linear_agent/task3/seed_sweep*.sh and
the main training runs) -- does not retrain anything itself. Missing files
are skipped with a warning rather than raising, so a partial results/ tree
(e.g. after the disk-hygiene cleanup described in the report) still produces
whatever figures it can.

Produces 3 figures in results/linear_agent/task3/:
  1. seed_sweep_consistency.png   -- kills/suicides/score across the 5 seeds
                                      (401-405) behind the final validated
                                      coin_collector_agent configuration
  2. fix_attempts_comparison.png  -- baseline vs. every attempted fix vs. the
                                      eventual winner, kills/suicides/score
  3. learning_curve.png           -- smoothed kills/suicides/score per round
                                      over the full 5000-round seed-401
                                      training run
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "results" / "linear_agent" / "task3"
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


def load_eval(rel_path: str) -> dict | None:
    path = REPO_ROOT / rel_path
    if not path.is_file():
        print(f"  [skip] missing: {rel_path}")
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("by_agent", data)


def eval_rounds(rel_path: str) -> int:
    path = REPO_ROOT / rel_path
    with open(path) as f:
        data = json.load(f)
    return data.get("by_agent", data).get("linear_agent", {}).get("rounds", 200)


# ── Figure 1: 5-seed consistency (coin_collector_agent, final config) ──────
def plot_seed_sweep_consistency():
    seeds = [401, 402, 403, 404, 405]
    kills, suicides, scores, opp_scores = [], [], [], []
    for seed in seeds:
        ba = load_eval(
            f"results/archive_task3_collector_final_seed_sweep/seed_{seed}/"
            f"task3_collector_finalseed_{seed}_eval.json"
        )
        if ba is None:
            print("  [abort] seed sweep figure needs all 5 seeds")
            return
        la = ba["linear_agent"]
        rounds = la["rounds"]
        kills.append(100 * la["kills"] / rounds)
        suicides.append(100 * la["suicides"] / rounds)
        scores.append(la["score"])
        opp_scores.append(ba["coin_collector_agent"]["score"])

    x = np.arange(len(seeds))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.bar(x - width / 2, kills, width, label="Kill rate", color="#4CAF50")
    ax.bar(x + width / 2, suicides, width, label="Suicide rate", color="#E91E63")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("% of 200 evaluation rounds")
    ax.set_title("Kill / suicide rate across 5 seeds")
    ax.legend()

    ax = axes[1]
    ax.bar(x - width / 2, scores, width, label="linear_agent (ours)", color="#2196F3")
    ax.bar(x + width / 2, opp_scores, width, label="coin_collector_agent", color="#9E9E9E")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("Score (200-round eval)")
    ax.set_title("Score across 5 seeds")
    ax.legend()

    fig.suptitle("Task 3: seed-to-seed consistency, final coin_collector_agent configuration")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "seed_sweep_consistency.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote seed_sweep_consistency.png")


# ── Figure 2: baseline vs. every attempted fix vs. the winner ──────────────
def plot_fix_attempts_comparison():
    configs = [
        ("Baseline\n(synth. escape)", "results/archive_task3_collector_synthescape/task3_collector_synthescape_5k_eval.json"),
        ("Margin feature\n(FAILED)", "results/archive_task3_collector_margin_FAILED/task3_collector_margin_5k_eval.json"),
        ("Predictive hit\n(FAILED)", "results/archive_task3_collector_predictivehit_FAILED/task3_collector_predictivehit_5k_eval.json"),
        ("Full safety veto\n(FAILED)", "results/archive_task3_collector_safeveto_FULL_worse_stalemates/task3_collector_safeveto_5k_eval.json"),
        ("Escape-only veto\n(FAILED)", "results/archive_task3_collector_escapeveto_ONLY_FAILED/task3_collector_escapeveto_5k_eval.json"),
        ("Final (seed 401)\nWINNER", "results/archive_task3_collector_final_seed_sweep/seed_401/task3_collector_finalseed_401_eval.json"),
    ]

    labels, kills, suicides, scores = [], [], [], []
    for label, rel_path in configs:
        ba = load_eval(rel_path)
        if ba is None:
            continue
        la = ba["linear_agent"]
        rounds = la["rounds"]
        labels.append(label)
        kills.append(100 * la["kills"] / rounds)
        suicides.append(100 * la["suicides"] / rounds)
        scores.append(la["score"])

    x = np.arange(len(labels))
    width = 0.35
    colors = ["#9E9E9E"] + ["#E91E63"] * (len(labels) - 2) + ["#4CAF50"] if len(labels) >= 2 else ["#9E9E9E"] * len(labels)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax = axes[0]
    bars1 = ax.bar(x - width / 2, kills, width, label="Kill rate", color="#2196F3")
    bars2 = ax.bar(x + width / 2, suicides, width, label="Suicide rate", color="#E91E63")
    ax.set_ylabel("% of 200 evaluation rounds")
    ax.set_title("Task 3: coin_collector_agent -- every attempted fix, kill/suicide rate")
    ax.legend()

    ax = axes[1]
    ax.bar(x, scores, color=colors)
    ax.set_ylabel("Score (200-round eval)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("Score per configuration (grey = pre-attempt baseline, red = failed, green = shipped)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fix_attempts_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fix_attempts_comparison.png")


# ── Figure 3: learning curve, seed 401 (5000 training rounds) ──────────────
def plot_learning_curve():
    path = REPO_ROOT / "results/archive_task3_collector_final_seed_sweep/seed_401/task3_collector_finalseed_401_train.json"
    if not path.is_file():
        print(f"  [skip] missing: {path}")
        return
    with open(path) as f:
        data = json.load(f)
    by_round = data["by_round"]

    # by_round keys look like "Round 0042 (...)" -- sort numerically, not lexically
    def round_num(key: str) -> int:
        return int(key.split()[1])

    rows = sorted(by_round.items(), key=lambda kv: round_num(kv[0]))
    kills = np.array([r[1]["kills"] for r in rows], dtype=float)
    suicides = np.array([r[1]["suicides"] for r in rows], dtype=float)
    coins = np.array([r[1]["coins"] for r in rows], dtype=float)

    def smooth(arr, win):
        if len(arr) < win:
            return arr
        kernel = np.ones(win) / win
        return np.convolve(arr, kernel, mode="valid")

    rounds = np.arange(len(kills))
    smoothed_rounds = rounds[SMOOTH_WIN - 1:] if len(rounds) >= SMOOTH_WIN else rounds

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(smoothed_rounds, smooth(kills, SMOOTH_WIN), label="kills/round", color="#4CAF50")
    ax.plot(smoothed_rounds, smooth(suicides, SMOOTH_WIN), label="suicides/round", color="#E91E63")
    ax.plot(smoothed_rounds, smooth(coins, SMOOTH_WIN) / 9, label="coins/round (/9, normalized)", color="#2196F3", alpha=0.6)
    ax.set_xlabel("Training round")
    ax.set_ylabel(f"Rolling mean (window={SMOOTH_WIN})")
    ax.set_title("Task 3: seed-401 training curve vs. coin_collector_agent (5000 rounds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "learning_curve.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote learning_curve.png")


# ── Figure 4: weight evolution over training (needs a per-step CSV) ────────
# See task4/plot_task4.py's plot_weight_evolution for why this needs a
# regenerated run rather than an archived one: the per-step training_log.csv
# is the single largest contributor to results/ disk usage over a long run
# and was not kept for the original seed-401 sweep. Regenerate with:
# AGENT_SEED=401 uv run main.py play --agents linear_agent coin_collector_agent
# --train 1 --scenario classic --no-gui --n-rounds 5000
# then copy agent_code/linear_agent/metrics/training_log.csv here. Note this
# necessarily uses the CURRENT codebase, which (unlike the original seed-401
# training run) also includes the later, Task-4-derived
# _adjacent_opponent_can_bomb veto -- active unconditionally regardless of
# opponent, so it does affect this run too. The qualitative weight
# trajectories are still representative of how Task 3's own features
# develop; exact final eval numbers from a literal rerun may differ slightly
# from the seed-401 table entries reported elsewhere in this chapter, which
# come from the original (pre-veto) archived run.
WEIGHT_COLUMNS = ["w_bomb", "w_moves_to_opponent", "w_escape_correct_move", "w_bomb_hits_opponent", "w_escape_exists"]


def plot_weight_evolution():
    path = REPO_ROOT / "results/archive_task3_collector_weightevo/training_log.csv"
    if not path.is_file():
        print(f"  [skip] missing: {path} (see plot_weight_evolution's docstring to regenerate)")
        return

    import csv as csv_module

    last_row_per_round = {}
    with open(path) as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            last_row_per_round[int(row["round"])] = row
    rounds = sorted(last_row_per_round.keys())

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#E91E63", "#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
    for label, color in zip(WEIGHT_COLUMNS, colors):
        values = [float(last_row_per_round[r][label]) for r in rounds]
        ax.plot(rounds, values, label=label, color=color, linewidth=1.3)

    ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Training round")
    ax.set_ylabel("Weight value")
    ax.set_title("Task 3: weight evolution, coin_collector_agent configuration (5000 rounds)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "weight_evolution.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote weight_evolution.png")


if __name__ == "__main__":
    print("Task 3 figures:")
    plot_seed_sweep_consistency()
    plot_fix_attempts_comparison()
    plot_learning_curve()
    plot_weight_evolution()
    print("Done ->", OUT_DIR)
