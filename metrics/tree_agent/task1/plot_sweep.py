

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]


SWEEP_DIR   = REPO_ROOT / "results" / "tree_agent" / "task1"
OUT_DIR     = REPO_ROOT / "results" / "tree_agent" / "task1"
SMOOTH_WIN  = 50
FINAL_WIN   = 100
N_COINS_MAX = 50

NE_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
NE_STYLES = ["-", "--", "-.", ":"]

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



def load_episodes(csv_path: str | Path) -> dict:

    episodes = defaultdict(lambda: {
        "sparse": 0.0,
        "shaped": 0.0,
        "invalid": 0,
        "steps": 0,
        "best_q_sum": 0.0,
        "best_q_n": 0,
    })
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = episodes[int(row["round"])]
            ep["sparse"]  += float(row["sparse_reward"])
            ep["shaped"]  += float(row["shaped_reward"])
            ep["invalid"] += 0 if int(row["valid"]) else 1
            ep["steps"]    = max(ep["steps"], int(row["step"]))
            if row["best_q"] != "":
                ep["best_q_sum"] += float(row["best_q"])
                ep["best_q_n"]   += 1
    return episodes



def load_all_results(sweep_dir: Path) -> list[dict]:

    results = []
    pattern = re.compile(r"log_ri(\d+)_ne(\d+)\.csv")

    for csv_file in sorted(sweep_dir.glob("log_ri*_ne*.csv")):
        m = pattern.match(csv_file.name)
        if not m:
            continue
        refit_interval = int(m.group(1))
        n_estimators = int(m.group(2))

        episodes = load_episodes(csv_file)
        rounds = sorted(episodes)

        results.append({
            "refit_interval": refit_interval,
            "n_estimators":   n_estimators,
            "rounds":         rounds,
            "coins":   [round(max(episodes[r]["sparse"], 0)) for r in rounds],
            "sparse":  [episodes[r]["sparse"] for r in rounds],
            "shaped":  [episodes[r]["shaped"] for r in rounds],
            "invalid": [episodes[r]["invalid"] for r in rounds],
            "steps":   [episodes[r]["steps"] for r in rounds],
            "best_q":  [
                episodes[r]["best_q_sum"] / episodes[r]["best_q_n"]
                if episodes[r]["best_q_n"] > 0 else np.nan
                for r in rounds
            ],
        })
        print(f"  Loaded REFIT_INTERVAL={refit_interval}  N_ESTIMATORS={n_estimators}  ({len(rounds)} rounds)")

    return results



def _smooth(values: list, window: int) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if np.isnan(arr).any():
        out = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            lo = max(0, i - window // 2)
            hi = min(len(arr), i + window // 2 + 1)
            window_vals = arr[lo:hi]
            valid = window_vals[~np.isnan(window_vals)]
            if len(valid) > 0:
                out[i] = valid.mean()
        return out
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def _fmt_ep(x, _):
    if x < 1000:
        return str(int(x))
    return f"{x/1000:g}k"


def _unique_sorted(results, key):
    return sorted(set(r[key] for r in results))


def _find(results, refit_interval, n_estimators):
    return next(
        (r for r in results if r["refit_interval"] == refit_interval and r["n_estimators"] == n_estimators),
        None
    )



def plot_learning_curves(
    results: list[dict],
    metric: str,
    ylabel: str,
    title: str,
    fname: str,
    hline: float | None = None,
    hline_label: str = "",
):

    refit_intervals = _unique_sorted(results, "refit_interval")
    n_estimators_vals = _unique_sorted(results, "n_estimators")

    n_cols = 2
    n_rows = (len(refit_intervals) + 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6.5 * n_cols, 4 * n_rows),
        sharey=True, sharex=True,
    )
    axes = np.array(axes).flatten()

    for ri_i, refit_interval in enumerate(refit_intervals):
        ax = axes[ri_i]
        for ne_i, n_estimators in enumerate(n_estimators_vals):
            run = _find(results, refit_interval, n_estimators)
            if run is None:
                continue
            smoothed = _smooth(run[metric], SMOOTH_WIN)
            eps = np.arange(1, len(smoothed) + 1)
            ax.plot(
                eps, smoothed,
                color=NE_COLORS[ne_i % len(NE_COLORS)],
                linestyle=NE_STYLES[ne_i % len(NE_STYLES)],
                linewidth=1.6,
                label=f"N_ESTIMATORS={n_estimators}",
            )

        if hline is not None:
            ax.axhline(hline, color="gray", linestyle="--",
                       linewidth=1.0, label=hline_label)

        ax.set_title(f"REFIT_INTERVAL = {refit_interval}", pad=6)
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.legend(loc="lower right", framealpha=0.6)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_ep))
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    for ax in axes[len(refit_intervals):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    out = OUT_DIR / fname
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")



def plot_heatmap(
    results: list[dict],
    value_fn,
    title: str,
    cbar_label: str,
    fname: str,
    cmap: str = "YlGn",
    fmt: str = ".1f",
):
    refit_intervals = _unique_sorted(results, "refit_interval")
    n_estimators_vals = _unique_sorted(results, "n_estimators")

    matrix = np.full((len(refit_intervals), len(n_estimators_vals)), np.nan)
    for i, refit_interval in enumerate(refit_intervals):
        for j, n_estimators in enumerate(n_estimators_vals):
            run = _find(results, refit_interval, n_estimators)
            if run is not None:
                matrix[i, j] = value_fn(run)

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(n_estimators_vals)))
    ax.set_xticklabels([str(n) for n in n_estimators_vals])
    ax.set_yticks(range(len(refit_intervals)))
    ax.set_yticklabels([str(r) for r in refit_intervals])
    ax.set_xlabel("N_ESTIMATORS")
    ax.set_ylabel("REFIT_INTERVAL")
    ax.set_title(title)

    vmax = np.nanmax(matrix)
    for i in range(len(refit_intervals)):
        for j in range(len(n_estimators_vals)):
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
    print(f"Saved -> {out}")



def plot_refit_interval_comparison(results: list[dict]):

    n_estimators_vals = _unique_sorted(results, "n_estimators")
    refit_intervals = _unique_sorted(results, "refit_interval")

    best_n_estimators = max(
        n_estimators_vals,
        key=lambda ne: np.mean([
            np.mean(r["coins"][-FINAL_WIN:])
            for r in results if r["n_estimators"] == ne
        ]),
    )

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for idx, refit_interval in enumerate(refit_intervals):
        run = _find(results, refit_interval, best_n_estimators)
        if run is None:
            continue
        smoothed = _smooth(run["coins"], SMOOTH_WIN)
        eps = np.arange(1, len(smoothed) + 1)
        final_avg = np.mean(run["coins"][-FINAL_WIN:])
        ax.plot(
            eps, smoothed,
            color=NE_COLORS[idx % len(NE_COLORS)],
            linestyle=NE_STYLES[idx % len(NE_STYLES)],
            linewidth=2.0,
            label=f"REFIT_INTERVAL={refit_interval}  (final avg={final_avg:.1f})",
        )

    ax.axhline(N_COINS_MAX, color="gray", linestyle="--",
               linewidth=1.0, label=f"Max ({N_COINS_MAX} coins)")
    ax.set_title(f"Effect of REFIT_INTERVAL   (N_ESTIMATORS={best_n_estimators} fixed)")
    ax.set_xlabel("Round")
    ax.set_ylabel(f"Coins collected (smoothed, w={SMOOTH_WIN})")
    ax.legend(framealpha=0.7)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_ep))

    fig.tight_layout()
    out = OUT_DIR / "refit_interval_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")



def plot_best_q_evolution(results: list[dict]):

    refit_intervals = _unique_sorted(results, "refit_interval")
    n_estimators_vals = _unique_sorted(results, "n_estimators")

    fig, axes = plt.subplots(
        len(refit_intervals) // 2, 2,
        figsize=(13, 4 * (len(refit_intervals) // 2)),
        sharey=False, sharex=True,
    )
    axes = np.array(axes).flatten()

    for ri_i, refit_interval in enumerate(refit_intervals):
        ax = axes[ri_i]
        for ne_i, n_estimators in enumerate(n_estimators_vals):
            run = _find(results, refit_interval, n_estimators)
            if run is None:
                continue
            smoothed = _smooth(run["best_q"], SMOOTH_WIN)
            eps = np.arange(1, len(smoothed) + 1)
            ax.plot(
                eps, smoothed,
                color=NE_COLORS[ne_i % len(NE_COLORS)],
                linestyle=NE_STYLES[ne_i % len(NE_STYLES)],
                linewidth=1.6,
                label=f"N_ESTIMATORS={n_estimators}",
            )

        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
        ax.set_title(f"REFIT_INTERVAL = {refit_interval}", pad=6)
        ax.set_xlabel("Round")
        ax.set_ylabel("mean best_q")
        ax.legend(loc="lower right", framealpha=0.6)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_ep))
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle(

        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    out = OUT_DIR / "best_q_evolution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")



def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading CSVs from {SWEEP_DIR} ...\n")
    results = load_all_results(SWEEP_DIR)

    if not results:
        print(f"No CSV files found in {SWEEP_DIR}.")
        print("Expected filenames like: log_ri50_ne50.csv")
        return

    print(f"\nLoaded {len(results)} configs. Generating plots ...\n")

    plot_learning_curves(
        results,
        metric="coins",
        ylabel=f"Coins collected (smoothed, w={SMOOTH_WIN})",
        title="Task 1 (tree_agent) - Coins collected per round across hyperparameters",
        fname="learning_curves_coins.png",
        hline=N_COINS_MAX,
        hline_label=f"Max ({N_COINS_MAX} coins)",
    )

    plot_learning_curves(
        results,
        metric="sparse",
        ylabel=f"Sparse reward (smoothed, w={SMOOTH_WIN})",
        title="Task 1 (tree_agent) - Sparse reward per round across hyperparameters",
        fname="learning_curves_reward.png",
    )

    plot_heatmap(
        results,
        value_fn=lambda r: np.mean(r["coins"][-FINAL_WIN:]),
        title=f"Final avg coins collected  (last {FINAL_WIN} rounds)",
        cbar_label="Avg coins / round",
        fname="heatmap_final_coins.png",
        cmap="YlGn",
    )

    plot_heatmap(
        results,
        value_fn=lambda r: np.mean(r["invalid"][-FINAL_WIN:]),
        title=f"Avg invalid actions  (last {FINAL_WIN} rounds)",
        cbar_label="Avg invalid actions / round",
        fname="heatmap_invalid_actions.png",
        cmap="OrRd_r",
    )

    plot_refit_interval_comparison(results)

    plot_best_q_evolution(results)

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
