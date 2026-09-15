"""
plot_task4.py -- Task 4 (rule_based_agent) result figures
================================================================================
Run from anywhere after cloning (self-locating, resolves the repo root from
its own file path):

    uv run metrics/linear_agent/task4/plot_task4.py

Reads exclusively from already-archived results/archive_task4_*/ and
results/task4_ffa_eval/ JSON files -- does not retrain anything itself.
Missing files are skipped with a warning.

Produces 4 figures in results/task4/:
  1. seed_sweep_consistency.png   -- kills/suicides/score across the 5 seeds
                                      (501-505) behind the shipped model
  2. fix_attempts_comparison.png  -- baseline vs. every attempted fix vs. the
                                      eventual winner (seed 505), kills/
                                      suicides/score
  3. ffa_comparison.png           -- shipped model vs. 3x rule_based_agent in
                                      the 4-player free-for-all generalization
                                      test
  4. learning_curve.png           -- smoothed kills/suicides/score per round
                                      over the full 5000-round seed-505
                                      training run
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "results" / "task4"
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


# ── Figure 1: 5-seed consistency (rule_based_agent, shipped config) ────────
def plot_seed_sweep_consistency():
    seeds = [501, 502, 503, 504, 505]
    kills, suicides, scores, opp_scores = [], [], [], []
    for seed in seeds:
        ba = load_eval(f"results/archive_task4_rulebased_seed_sweep/seed_{seed}/eval.json")
        if ba is None:
            print("  [abort] seed sweep figure needs all 5 seeds")
            return
        la = ba["linear_agent"]
        rounds = la["rounds"]
        kills.append(100 * la["kills"] / rounds)
        suicides.append(100 * la["suicides"] / rounds)
        scores.append(la["score"])
        opp_scores.append(ba["rule_based_agent"]["score"])

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
    ax.bar(x + width / 2, opp_scores, width, label="rule_based_agent", color="#9E9E9E")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("Score (200-round eval)")
    ax.set_title("Score across 5 seeds")
    ax.legend()

    fig.suptitle("Task 4: seed-to-seed consistency, adjacent-opponent-can-bomb veto vs. rule_based_agent")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "seed_sweep_consistency.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote seed_sweep_consistency.png")


# ── Figure 2: baseline vs. every attempted fix vs. the winner ──────────────
def plot_fix_attempts_comparison():
    configs = [
        ("Baseline\n(no fix)", "results/archive_task4_rulebased_5k_baseline/task4_rulebased_5k_eval.json"),
        ("Retaliation-aware\nescape only", "results/archive_task4_rulebased_retaliation/task4_rulebased_retaliation_5k_eval.json"),
        ("Adjacent-bomb veto\nWINNER (single run)", "results/archive_task4_rulebased_hardveto/task4_rulebased_hardveto_5k_eval.json"),
        ("Escape-override\nretry (FAILED)", "results/archive_task4_escape_override_retry_FAILED/vs_rulebased_eval.json"),
        ("WAIT veto\n(MIXED, reverted)", "results/archive_task4_wait_veto_MIXED_reverted/vs_rulebased_eval.json"),
        ("Final shipped\n(seed 505)", "results/archive_task4_rulebased_seed_sweep/seed_505/eval.json"),
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
    bar_colors = ["#9E9E9E", "#E91E63", "#4CAF50", "#E91E63", "#FF9800", "#4CAF50"][: len(labels)]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax = axes[0]
    ax.bar(x - width / 2, kills, width, label="Kill rate", color="#2196F3")
    ax.bar(x + width / 2, suicides, width, label="Suicide rate", color="#E91E63")
    ax.set_ylabel("% of 200 evaluation rounds")
    ax.set_title("Task 4: rule_based_agent -- every attempted fix, kill/suicide rate")
    ax.legend()

    ax = axes[1]
    ax.bar(x, scores, color=bar_colors)
    ax.set_ylabel("Score (200-round eval)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_title(
        "Score per configuration (grey = baseline, red = failed, orange = mixed result, green = shipped)"
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fix_attempts_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fix_attempts_comparison.png")


# ── Figure 3: 4-player free-for-all generalization test ────────────────────
def plot_ffa_comparison():
    ba = load_eval("results/task4_ffa_eval/vs_3x_rulebased.json")
    if ba is None:
        return

    agents = ["linear_agent", "rule_based_agent_0", "rule_based_agent_1", "rule_based_agent_2"]
    display_labels = ["ours", "rule_based\n#1", "rule_based\n#2", "rule_based\n#3"]
    kills = [100 * ba[a]["kills"] / ba[a]["rounds"] for a in agents]
    suicides = [100 * ba[a]["suicides"] / ba[a]["rounds"] for a in agents]
    scores = [ba[a]["score"] for a in agents]
    colors = ["#2196F3", "#9E9E9E", "#9E9E9E", "#9E9E9E"]
    x = np.arange(len(agents))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))

    axes[0].bar(x, scores, color=colors, width=0.65)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Score")

    axes[1].bar(x, kills, color=colors, width=0.65)
    axes[1].set_ylabel("% of 200 rounds")
    axes[1].set_title("Kill rate")

    axes[2].bar(x, suicides, color=colors, width=0.65)
    axes[2].set_ylabel("% of 200 rounds")
    axes[2].set_title("Suicide rate")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(display_labels, fontsize=9.5)

    fig.suptitle("Task 4: 4-player free-for-all, shipped model (no retraining) vs. 3x rule_based_agent")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ffa_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote ffa_comparison.png")


# ── Figure 4: learning curve, seed 505 (5000 training rounds) ──────────────
def plot_learning_curve():
    path = REPO_ROOT / "results/archive_task4_rulebased_seed_sweep/seed_505/train.json"
    if not path.is_file():
        print(f"  [skip] missing: {path}")
        return
    with open(path) as f:
        data = json.load(f)
    by_round = data["by_round"]

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
    ax.set_title("Task 4: seed-505 training curve vs. rule_based_agent (5000 rounds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "learning_curve.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote learning_curve.png")


# ── Figure 5: weight evolution over training (needs a per-step CSV) ────────
# Unlike the other figures, this one needs the per-step training_log.csv
# (round, step, ..., w_coin, w_valid, w_wait, w_bomb, ..., w_moves_to_opponent,
# w_bomb_hits_opponent), which is NOT kept in most archives (it is the single
# largest contributor to results/ disk usage over a long run, see the
# disk-hygiene notes in the report, so only a few runs' copies survive).
# Regenerate it with: AGENT_SEED=505 uv run main.py play --agents linear_agent
# rule_based_agent --train 1 --scenario classic --no-gui --n-rounds 5000
# (reproduces the exact shipped configuration), then copy
# agent_code/linear_agent/metrics/training_log.csv here before it's
# overwritten by the next training run.
WEIGHT_COLUMNS = ["w_bomb", "w_moves_to_opponent", "w_escape_correct_move", "w_bomb_hits_opponent", "w_escape_exists"]


def plot_weight_evolution():
    path = REPO_ROOT / "results/archive_task4_rulebased_weightevo/training_log.csv"
    if not path.is_file():
        print(f"  [skip] missing: {path} (see plot_weight_evolution's docstring to regenerate)")
        return

    import csv as csv_module

    rounds, last_row_per_round = [], {}
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
    ax.set_title("Task 4: weight evolution, seed-505 shipped configuration (5000 rounds)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "weight_evolution.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote weight_evolution.png")


if __name__ == "__main__":
    print("Task 4 figures:")
    plot_seed_sweep_consistency()
    plot_fix_attempts_comparison()
    plot_ffa_comparison()
    plot_learning_curve()
    plot_weight_evolution()
    print("Done ->", OUT_DIR)
