"""
plot_task2.py -- Task 2 diagnostic figures for report2_body.typ
=================================================================
Run from the repo root:

    python plot_task2.py

Reads checkpoint/eval artifacts already saved under results/ (from the
decaying-alpha run and the two CRATE_HIT_BONUS attempts) and produces three
figures in results/linear_agent/task2/:

  1. diagnosis_before_after.png   -- commit-gap & w_crate_hit_if_bomb across
                                      all checkpoints, decaying-alpha (broken)
                                      vs CRATE_HIT_BONUS=5.0 50k (fixed)
  2. eval_distribution_before_after.png -- pure-greedy coin distributions,
                                      CRATE_HIT_BONUS=1.5 (5k vs 20k, contradict
                                      each other) vs CRATE_HIT_BONUS=5.0
                                      (5k/20k/50k, all agree)
  3. config_comparison_bar.png    -- coins/round summary bar chart across
                                      every configuration tested
"""

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("results/linear_agent/task2")
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

BOMB_COMMIT = np.array([0, 1, 0, 1, 0, 1, 1, 0, 0])
WAIT_COMMIT = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0])

ACCENT = "#1f4e79"
WARN = "#c0392b"
GOOD = "#2e8b57"
GREY = "#8a8a8a"


def load_checkpoint_trend(ckpt_dir: Path):
    ckpts = sorted(ckpt_dir.glob("weights_round_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    rounds, gaps, crate_w = [], [], []
    for p in ckpts:
        r = int(p.stem.split("_")[-1])
        with open(p, "rb") as f:
            w = pickle.load(f)
        w = np.asarray(w)
        gap = float(np.dot(w, BOMB_COMMIT) - np.dot(w, WAIT_COMMIT))
        rounds.append(r)
        gaps.append(gap)
        crate_w.append(float(w[6]))
    return np.array(rounds), np.array(gaps), np.array(crate_w)


def load_eval_coins(eval_json_path: Path):
    with open(eval_json_path) as f:
        d = json.load(f)
    return np.array([v["coins"] for v in d["by_round"].values()])


# ── Figure 1: before/after diagnosis (commit-gap & w_crate_hit) ─────────────

before_dir = Path("results/archive_decaying_alpha_50k_broken_crate_hit/checkpoints")
after_dir = Path("agent_code/linear_agent/checkpoints")  # live: final CRATE_HIT_BONUS=5.0 50k run

r_before, gap_before, crate_before = load_checkpoint_trend(before_dir)
r_after, gap_after, crate_after = load_checkpoint_trend(after_dir)

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col")

axes[0, 0].axhline(0, color=GREY, lw=1, ls="--")
axes[0, 0].plot(r_before, gap_before, "o-", color=WARN, ms=4)
axes[0, 0].set_title("Before: decaying $\\alpha$ (CRATE_HIT_BONUS absent)")
axes[0, 0].set_ylabel("Commit-gap\n$Q$(BOMB, hits crate) $-$ $Q$(WAIT)")

axes[0, 1].axhline(0, color=GREY, lw=1, ls="--")
axes[0, 1].plot(r_after, gap_after, "o-", color=GOOD, ms=4)
axes[0, 1].set_title("After: CRATE_HIT_BONUS = 5.0")

ylim = (min(gap_before.min(), gap_after.min()) - 0.15, max(gap_before.max(), gap_after.max()) + 0.15)
axes[0, 0].set_ylim(*ylim)
axes[0, 1].set_ylim(*ylim)

axes[1, 0].axhline(0, color=GREY, lw=1, ls="--")
axes[1, 0].plot(r_before, crate_before, "o-", color=WARN, ms=4)
axes[1, 0].set_ylabel("$w$_crate_hit_if_bomb")
axes[1, 0].set_xlabel("Training round")

axes[1, 1].axhline(0, color=GREY, lw=1, ls="--")
axes[1, 1].plot(r_after, crate_after, "o-", color=GOOD, ms=4)
axes[1, 1].set_xlabel("Training round")

ylim2 = (min(crate_before.min(), crate_after.min()) - 0.3, max(crate_before.max(), crate_after.max()) + 0.3)
axes[1, 0].set_ylim(*ylim2)
axes[1, 1].set_ylim(*ylim2)

for ax in axes.flat:
    ax.grid(alpha=0.25)

fig.suptitle("Resolving the crate-bombing decision: 50,000-round checkpoint trajectories", y=1.01, fontsize=13)
fig.tight_layout()
fig.savefig(OUT_DIR / "diagnosis_before_after.png", bbox_inches="tight")
plt.close(fig)
print("saved diagnosis_before_after.png")


# ── Figure 2: eval coin distributions, unresolved vs resolved ──────────────

bonus15_5k = load_eval_coins(Path("results/archive_crate_hit_bonus_5k/crate_hit_bonus_5k_eval.json"))
bonus15_20k = load_eval_coins(Path("results/archive_crate_hit_bonus_20k/crate_hit_bonus_20k_eval.json"))

bonus50_5k = load_eval_coins(Path("results/archive_crate_hit_bonus5_5k/crate_hit_bonus5_5k_eval.json"))
bonus50_20k = load_eval_coins(Path("results/archive_crate_hit_bonus5_20k/crate_hit_bonus5_20k_eval.json"))
bonus50_50k = load_eval_coins(Path("results/crate_hit_bonus5_50k_eval.json"))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

bins = np.arange(0, 52, 2)
axes[0].hist(bonus15_5k, bins=bins, alpha=0.65, color=ACCENT, label=f"5,000 rounds (mean {bonus15_5k.mean():.1f})")
axes[0].hist(bonus15_20k, bins=bins, alpha=0.65, color=WARN, label=f"20,000 rounds (mean {bonus15_20k.mean():.1f})")
axes[0].set_title("CRATE_HIT_BONUS = 1.5\n(5k and 20k contradict each other)")
axes[0].set_xlabel("Coins collected per round (200-round eval)")
axes[0].set_ylabel("Number of rounds")
axes[0].legend(loc="upper center")

axes[1].hist(bonus50_5k, bins=bins, alpha=0.55, color="#4CAF50", label=f"5,000 rounds (mean {bonus50_5k.mean():.1f})")
axes[1].hist(bonus50_20k, bins=bins, alpha=0.55, color="#2196F3", label=f"20,000 rounds (mean {bonus50_20k.mean():.1f})")
axes[1].hist(bonus50_50k, bins=bins, alpha=0.55, color="#9C27B0", label=f"50,000 rounds (mean {bonus50_50k.mean():.1f})")
axes[1].set_title("CRATE_HIT_BONUS = 5.0\n(5k, 20k, 50k all agree)")
axes[1].set_xlabel("Coins collected per round (200-round eval)")
axes[1].legend(loc="upper left")

for ax in axes:
    ax.grid(alpha=0.25)

fig.suptitle("Pure-greedy evaluation: short-run snapshots vs. long-run reality", y=1.03, fontsize=13)
fig.tight_layout()
fig.savefig(OUT_DIR / "eval_distribution_before_after.png", bbox_inches="tight")
plt.close(fig)
print("saved eval_distribution_before_after.png")


# ── Figure 3: coins/round summary bar chart across every config tested ─────

decaying_alpha_50k = load_eval_coins(Path("results/archive_decaying_alpha_50k_broken_crate_hit/decaying_alpha_50k_eval.json"))

configs = [
    ("Decaying $\\alpha$\n(50k)", decaying_alpha_50k.mean(), WARN),
    ("Bonus=1.5\n(5k)", bonus15_5k.mean(), "#e67e22"),
    ("Bonus=1.5\n(20k)", bonus15_20k.mean(), WARN),
    ("Bonus=5.0\n(5k)", bonus50_5k.mean(), GOOD),
    ("Bonus=5.0\n(20k)", bonus50_20k.mean(), GOOD),
    ("Bonus=5.0\n(50k, final)", bonus50_50k.mean(), ACCENT),
]

fig, ax = plt.subplots(figsize=(9, 4.5))
labels = [c[0] for c in configs]
values = [c[1] for c in configs]
colors = [c[2] for c in configs]
bars = ax.bar(labels, values, color=colors, width=0.6)
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=10)

ax.set_ylabel("Coins/round (200-round pure-greedy eval)")
ax.set_title("Every configuration tested for the bombing-decision fix")
ax.axhline(50, color=GREY, lw=1, ls=":")
ax.text(len(configs) - 0.4, 50.5, "max = 50", color=GREY, fontsize=9, ha="right")
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig(OUT_DIR / "config_comparison_bar.png", bbox_inches="tight")
plt.close(fig)
print("saved config_comparison_bar.png")

print("\nAll figures saved to", OUT_DIR)
