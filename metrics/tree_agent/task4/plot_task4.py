from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path("results/tree_agent/task4")
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


def load(rel_path: str, agent: str):
    d = json.load(open(RESULTS_DIR / rel_path))
    return d["by_agent"][agent]


# ── Figure 1: the adjacent-opponent veto fix, before vs. after ─────────────

def plot_fix():
    before = load("task4_spotcheck_pretask4.json", "tree_agent")
    after = load("task4_rulebased_eval.json", "tree_agent")

    configs = [
        ("Before\n(Task 3 model,\nno Task 4 fix)", before, WARN),
        ("After\n(retaliation escape\n+ adjacent veto)", after, GOOD),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    labels = [c[0] for c in configs]
    scores = [c[1]["score"] for c in configs]
    suicide_rates = [c[1]["suicides"] / c[1]["rounds"] * 100 for c in configs]
    colors = [c[2] for c in configs]

    ax = axes[0]
    bars = ax.bar(labels, scores, color=colors, width=0.6)
    for bar, v in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 15, f"{v}", ha="center", fontsize=10)
    ax.set_ylabel("Score (200-round pure-greedy eval)")
    ax.set_title("Our score")
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    bars = ax.bar(labels, suicide_rates, color=colors, width=0.6)
    for bar, v in zip(bars, suicide_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=10)
    ax.set_ylabel("Suicide rate")
    ax.set_title("Our suicide rate")
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("The adjacent-opponent veto fix, vs. rule_based_agent", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "adjacent_opponent_veto_fix.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote adjacent_opponent_veto_fix.png")


# ── Figure 2: 4-player free-for-all comparison ──────────────────────────────

def plot_ffa():
    d = json.load(open(RESULTS_DIR / "task4_ffa.json"))
    by_agent = d["by_agent"]

    order = ["tree_agent", "rule_based_agent_0", "rule_based_agent_1", "rule_based_agent_2"]
    labels = ["tree_agent\n(ours)", "rule_based\ncopy 1", "rule_based\ncopy 2", "rule_based\ncopy 3"]
    colors = [ACCENT, GREY, GREY, GREY]

    scores = [by_agent[a]["score"] for a in order]
    kill_rates = [by_agent[a]["kills"] / by_agent[a]["rounds"] * 100 for a in order]
    suicide_rates = [by_agent[a]["suicides"] / by_agent[a]["rounds"] * 100 for a in order]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    for ax, values, title, ylabel, fmt in [
        (axes[0], scores, "Score", "Score", "{:.0f}"),
        (axes[1], kill_rates, "Kill rate", "Kill rate (%)", "{:.1f}%"),
        (axes[2], suicide_rates, "Suicide rate", "Suicide rate (%)", "{:.1f}%"),
    ]:
        bars = ax.bar(labels, values, color=colors, width=0.6)
        vmax = max(values)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + vmax * 0.02, fmt.format(v), ha="center", fontsize=9)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.grid(alpha=0.25, axis="y")

    fig.suptitle("4-player free-for-all: tree_agent (1v1-trained) vs. 3x rule_based_agent", y=1.03, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ffa_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote ffa_comparison.png")


# ── Figure 3: 3-opponent regression check for the shipped model ────────────

def plot_regression_check():
    peaceful = load("task4_regression_peaceful.json", "tree_agent")
    collector = load("task4_regression_collector.json", "tree_agent")
    rulebased = load("task4_rulebased_eval.json", "tree_agent")

    # dedicated Task 3 models, for reference: how much (if any) is given up
    # by shipping ONE model trained only against rule_based_agent, instead
    # of each opponent's own dedicated model
    dedicated_peaceful = json.load(open(RESULTS_DIR.parent / "task3" / "task3_peaceful_eval_v2.json"))["by_agent"]["tree_agent"]
    dedicated_collector = json.load(open(RESULTS_DIR.parent / "task3" / "task3_collector_eval.json"))["by_agent"]["tree_agent"]

    opponents = ["peaceful_agent", "coin_collector_agent", "rule_based_agent"]
    shipped = [peaceful, collector, rulebased]
    dedicated = [dedicated_peaceful, dedicated_collector, None]

    x = range(len(opponents))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    for ax, key, title, ylabel, scale in [
        (axes[0], "score", "Score", "Score", 1),
        (axes[1], "kills", "Kill rate", "Kill rate (%)", 100),
        (axes[2], "suicides", "Suicide rate", "Suicide rate (%)", 100),
    ]:
        shipped_vals = [s[key] / (s["rounds"] if scale != 1 else 1) * scale for s in shipped]
        ax.bar([i - width / 2 for i in x], shipped_vals, width, color=ACCENT, label="Shipped (rule_based-trained)")

        dedicated_vals = []
        dedicated_x = []
        for i, d in enumerate(dedicated):
            if d is not None:
                dedicated_x.append(i + width / 2)
                dedicated_vals.append(d[key] / (d["rounds"] if scale != 1 else 1) * scale)
        ax.bar(dedicated_x, dedicated_vals, width, color=GOOD, label="Dedicated Task 3 model")

        vmax = max(shipped_vals + dedicated_vals)
        for xi, v in zip(x, shipped_vals):
            ax.text(xi - width / 2, v + vmax * 0.02, f"{v:.0f}" if scale == 1 else f"{v:.1f}%",
                    ha="center", fontsize=8)
        for xi, v in zip(dedicated_x, dedicated_vals):
            ax.text(xi, v + vmax * 0.02, f"{v:.0f}" if scale == 1 else f"{v:.1f}%",
                    ha="center", fontsize=8)

        ax.set_xticks(list(x))
        ax.set_xticklabels(["peaceful", "collector", "rule_based"], fontsize=9)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, axis="y")

    axes[2].legend(loc="upper right", fontsize=8)
    fig.suptitle("Regression check: one shipped model vs. all three opponents", y=1.03, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "regression_check.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote regression_check.png")


if __name__ == "__main__":
    plot_fix()
    plot_ffa()
    plot_regression_check()
