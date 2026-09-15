"""
plot_sweep.py  –  Comparison plots across all hyperparameter sweep CSVs
========================================================================
Run from the repo root after sweep.py has finished:

    python plot_sweep.py

Expects CSVs at:  results/sweep/log_a{alpha}_g{gamma}.csv
Saves figures to: results/sweep/

Produces 6 figures:
  1. learning_curves_coins.png     – smoothed coins/round per (α, γ)
  2. learning_curves_reward.png    – smoothed sparse reward per (α, γ)
  3. heatmap_final_coins.png       – α × γ heatmap, final avg coins
  4. heatmap_invalid_actions.png   – α × γ heatmap, final avg invalid actions
  5. alpha_comparison.png          – all α at the best γ on one axis
  6. weight_evolution.png          – w_moves_to_coin trajectory for all configs

Reuses load_episodes() logic from your existing plot_metrics.py.

Author: <your name here>
"""

from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

SWEEP_DIR    = Path("results/sweep")     # where sweep.py saved the CSVs
OUT_DIR      = Path("results/sweep")     # where to save figures
SMOOTH_WIN   = 50                        # rolling average window
FINAL_WIN    = 100                       # last N rounds for "final performance"
N_COINS_MAX  = 50                        # coin-heaven maximum

# Colours: one per γ value (4 values → 4 colours)
GAMMA_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
GAMMA_STYLES = ["-", "--", "-.", ":"]

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


# ── load_episodes: identical to your plot_metrics.py ─────────────────────────

def load_episodes(csv_path: str | Path) -> dict:
    """
    Aggregate per-step CSV rows into per-round dicts.
    Matches your existing plot_metrics.py logic exactly.
    """
    episodes = defaultdict(lambda: {
        "sparse": 0.0,
        "shaped": 0.0,
        "invalid": 0,
        "steps": 0,
        "w_moves_to_coin": 0.0,
        "w_valid": 0.0,
        "w_wait": 0.0,
        "w_bomb": 0.0,
    })
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = episodes[int(row["round"])]
            ep["sparse"]  += float(row["sparse_reward"])
            ep["shaped"]  += float(row["shaped_reward"])
            ep["invalid"] += 0 if int(row["valid"]) else 1
            ep["steps"]    = max(ep["steps"], int(row["step"]))
            ep["w_moves_to_coin"] = float(row["w_moves_to_coin"])
            ep["w_valid"]         = float(row["w_valid"])
            ep["w_wait"]          = float(row["w_wait"])
            ep["w_bomb"]          = float(row["w_bomb"])
    return episodes


# ── Load all sweep CSVs ───────────────────────────────────────────────────────

def load_all_results(sweep_dir: Path) -> list[dict]:
    """
    Find every log_a{alpha}_g{gamma}.csv in sweep_dir.
    Returns a list of result dicts, each with keys:
        alpha, gamma, rounds, coins, sparse, shaped,
        invalid, steps, w_moves_to_coin
    """
    results = []
    pattern = re.compile(r"log_a([\d.]+)_g([\d.]+)\.csv")

    for csv_file in sorted(sweep_dir.glob("log_a*_g*.csv")):
        m = pattern.match(csv_file.name)
        if not m:
            continue
        alpha = float(m.group(1))
        gamma = float(m.group(2))

        episodes = load_episodes(csv_file)
        rounds   = sorted(episodes)

        results.append({
            "alpha":          alpha,
            "gamma":          gamma,
            "rounds":         rounds,
            # Per-round metric lists
            "coins":          [round(max(episodes[r]["sparse"], 0))
                               for r in rounds],
            "sparse":         [episodes[r]["sparse"]        for r in rounds],
            "shaped":         [episodes[r]["shaped"]        for r in rounds],
            "invalid":        [episodes[r]["invalid"]       for r in rounds],
            "steps":          [episodes[r]["steps"]         for r in rounds],
            "w_moves_to_coin":[episodes[r]["w_moves_to_coin"] for r in rounds],
            "w_valid":        [episodes[r]["w_valid"]       for r in rounds],
            "w_wait":         [episodes[r]["w_wait"]        for r in rounds],
            "w_bomb":         [episodes[r]["w_bomb"]        for r in rounds],
        })
        print(f"  Loaded α={alpha}  γ={gamma}  ({len(rounds)} rounds)")

    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _smooth(values: list, window: int) -> np.ndarray:
    arr    = np.array(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def _fmt_ep(x, _):
    return f"{int(x/1000)}k" if x >= 1000 else str(int(x))


def _unique_sorted(results, key):
    return sorted(set(r[key] for r in results))


def _find(results, alpha, gamma):
    return next(
        (r for r in results if r["alpha"] == alpha and r["gamma"] == gamma),
        None
    )


# ── Plot 1 & 2: Learning curves ───────────────────────────────────────────────

def plot_learning_curves(
    results: list[dict],
    metric: str,
    ylabel: str,
    title: str,
    fname: str,
    hline: float | None = None,
    hline_label: str = "",
):
    """2×2 grid: one subplot per α, one line per γ."""
    alphas = _unique_sorted(results, "alpha")
    gammas = _unique_sorted(results, "gamma")

    n_cols = 2
    n_rows = (len(alphas) + 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6.5 * n_cols, 4 * n_rows),
        sharey=True, sharex=True,
    )
    axes = np.array(axes).flatten()

    for ai, alpha in enumerate(alphas):
        ax = axes[ai]
        for gi, gamma in enumerate(gammas):
            run = _find(results, alpha, gamma)
            if run is None:
                continue
            smoothed = _smooth(run[metric], SMOOTH_WIN)
            eps      = np.arange(1, len(smoothed) + 1)
            ax.plot(
                eps, smoothed,
                color    = GAMMA_COLORS[gi % len(GAMMA_COLORS)],
                linestyle= GAMMA_STYLES[gi % len(GAMMA_STYLES)],
                linewidth= 1.6,
                label    = f"γ={gamma:.2f}",
            )

        if hline is not None:
            ax.axhline(hline, color="gray", linestyle="--",
                       linewidth=1.0, label=hline_label)

        ax.set_title(f"α = {alpha}", pad=6)
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.legend(loc="lower right", framealpha=0.6)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_ep))
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    for ax in axes[len(alphas):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    out = OUT_DIR / fname
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── Plot 3 & 4: Heatmaps ─────────────────────────────────────────────────────

def plot_heatmap(
    results: list[dict],
    value_fn,
    title: str,
    cbar_label: str,
    fname: str,
    cmap: str = "YlGn",
    fmt: str = ".1f",
):
    alphas = _unique_sorted(results, "alpha")
    gammas = _unique_sorted(results, "gamma")

    matrix = np.full((len(alphas), len(gammas)), np.nan)
    for i, alpha in enumerate(alphas):
        for j, gamma in enumerate(gammas):
            run = _find(results, alpha, gamma)
            if run is not None:
                matrix[i, j] = value_fn(run)

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(gammas)))
    ax.set_xticklabels([f"{g:.2f}" for g in gammas])
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels([f"{a:.3f}" for a in alphas])
    ax.set_xlabel("Discount factor  γ")
    ax.set_ylabel("Learning rate  α")
    ax.set_title(title)

    vmax = np.nanmax(matrix)
    for i in range(len(alphas)):
        for j in range(len(gammas)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val < vmax * 0.6 else "black"
                ax.text(j, i, f"{val:{fmt}}",
                        ha="center", va="center",
                        color=color, fontsize=10, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    fig.tight_layout()

    out = OUT_DIR / fname
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── Plot 5: Alpha comparison ──────────────────────────────────────────────────

def plot_alpha_comparison(results: list[dict]):
    """All α values overlaid on one axis, fixing the best γ."""
    gammas = _unique_sorted(results, "gamma")
    alphas = _unique_sorted(results, "alpha")

    best_gamma = max(
        gammas,
        key=lambda g: np.mean([
            np.mean(r["coins"][-FINAL_WIN:])
            for r in results if r["gamma"] == g
        ]),
    )

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for idx, alpha in enumerate(alphas):
        run = _find(results, alpha, best_gamma)
        if run is None:
            continue
        smoothed  = _smooth(run["coins"], SMOOTH_WIN)
        eps       = np.arange(1, len(smoothed) + 1)
        final_avg = np.mean(run["coins"][-FINAL_WIN:])
        ax.plot(
            eps, smoothed,
            color    = GAMMA_COLORS[idx % len(GAMMA_COLORS)],
            linestyle= GAMMA_STYLES[idx % len(GAMMA_STYLES)],
            linewidth= 2.0,
            label    = f"α={alpha:.3f}  (final avg={final_avg:.1f})",
        )

    ax.axhline(N_COINS_MAX, color="gray", linestyle="--",
               linewidth=1.0, label=f"Max ({N_COINS_MAX} coins)")
    ax.set_title(f"Effect of learning rate α   (γ={best_gamma:.2f} fixed)")
    ax.set_xlabel("Round")
    ax.set_ylabel(f"Coins collected (smoothed, w={SMOOTH_WIN})")
    ax.legend(framealpha=0.7)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_ep))

    fig.tight_layout()
    out = OUT_DIR / "alpha_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── Plot 6: Weight evolution ──────────────────────────────────────────────────

def plot_weight_evolution(results: list[dict]):
    """
    w_moves_to_coin over training for every config on one axis.
    This is the most interpretable weight — it should rise as the
    agent learns that moving toward coins is good.
    """
    alphas = _unique_sorted(results, "alpha")
    gammas = _unique_sorted(results, "gamma")

    fig, axes = plt.subplots(
        len(alphas) // 2, 2,
        figsize=(13, 4 * (len(alphas) // 2)),
        sharey=False, sharex=True,
    )
    axes = np.array(axes).flatten()

    for ai, alpha in enumerate(alphas):
        ax = axes[ai]
        for gi, gamma in enumerate(gammas):
            run = _find(results, alpha, gamma)
            if run is None:
                continue
            smoothed = _smooth(run["w_moves_to_coin"], SMOOTH_WIN)
            eps      = np.arange(1, len(smoothed) + 1)
            ax.plot(
                eps, smoothed,
                color    = GAMMA_COLORS[gi % len(GAMMA_COLORS)],
                linestyle= GAMMA_STYLES[gi % len(GAMMA_STYLES)],
                linewidth= 1.6,
                label    = f"γ={gamma:.2f}",
            )

        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
        ax.set_title(f"α = {alpha}", pad=6)
        ax.set_xlabel("Round")
        ax.set_ylabel("w_moves_to_coin")
        ax.legend(loc="lower right", framealpha=0.6)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_ep))
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle(
        "Weight evolution: w_moves_to_coin across hyperparameters",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    out = OUT_DIR / "weight_evolution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading CSVs from {SWEEP_DIR} ...\n")
    results = load_all_results(SWEEP_DIR)

    if not results:
        print(f"No CSV files found in {SWEEP_DIR}.")
        print("Expected filenames like: log_a0.001_g0.99.csv")
        return

    print(f"\nLoaded {len(results)} configs. Generating plots ...\n")

    # 1. Learning curves: coins collected
    plot_learning_curves(
        results,
        metric      = "coins",
        ylabel      = f"Coins collected (smoothed, w={SMOOTH_WIN})",
        title       = "Task 1 – Coins collected per round across hyperparameters",
        fname       = "learning_curves_coins.png",
        hline       = N_COINS_MAX,
        hline_label = f"Max ({N_COINS_MAX} coins)",
    )

    # 2. Learning curves: sparse reward
    plot_learning_curves(
        results,
        metric = "sparse",
        ylabel = f"Sparse reward (smoothed, w={SMOOTH_WIN})",
        title  = "Task 1 – Sparse reward per round across hyperparameters",
        fname  = "learning_curves_reward.png",
    )

    # 3. Heatmap: final coins
    plot_heatmap(
        results,
        value_fn   = lambda r: np.mean(r["coins"][-FINAL_WIN:]),
        title      = f"Final avg coins collected  (last {FINAL_WIN} rounds)",
        cbar_label = "Avg coins / round",
        fname      = "heatmap_final_coins.png",
        cmap       = "YlGn",
    )

    # 4. Heatmap: invalid actions
    plot_heatmap(
        results,
        value_fn   = lambda r: np.mean(r["invalid"][-FINAL_WIN:]),
        title      = f"Avg invalid actions  (last {FINAL_WIN} rounds)",
        cbar_label = "Avg invalid actions / round",
        fname      = "heatmap_invalid_actions.png",
        cmap       = "OrRd_r",
    )

    # 5. Alpha comparison at best gamma
    plot_alpha_comparison(results)

    # 6. Weight evolution
    plot_weight_evolution(results)

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()