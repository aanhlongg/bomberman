"""
Runs `agent_pgk` for N episodes in headless training mode, then plots the
resulting learning curves (score and steps-per-episode) from
training_stats.csv.

Usage:
    python evaluate.py --n-rounds 200 --scenario coin-heaven
    python evaluate.py --n-rounds 200 --scenario coin-heaven --skip-run  # only re-plot existing CSV
    python evaluate.py --n-rounds 200 --scenario coin-heaven --fresh     # delete old stats first

Run this from the bomberman_rl root directory (same place you'd normally
run `main.py` from), since training_stats.csv is written there.
"""
import argparse
import os
import subprocess
import sys

import pandas as pd
import matplotlib.pyplot as plt

STATS_FILENAME = "training_stats.csv"


def run_training(n_rounds: int, scenario: str, agent: str):
    cmd = [
        sys.executable, "main.py", "play",
        "--no-gui",
        "--agents", agent,
        "--train", "1",
        "--scenario", scenario,
        "--n-rounds", str(n_rounds),
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Warning: main.py exited with code {result.returncode}", file=sys.stderr)


def plot_stats(stats_path: str, rolling_window: int, out_path: str, show: bool):
    if not os.path.isfile(stats_path):
        print(f"No {stats_path} found -- nothing to plot.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(stats_path)
    if df.empty:
        print(f"{stats_path} is empty -- nothing to plot.", file=sys.stderr)
        sys.exit(1)

    df["score_rolling"] = df["score"].rolling(rolling_window, min_periods=1).mean()
    df["steps_rolling"] = df["steps"].rolling(rolling_window, min_periods=1).mean()

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(df["round"], df["score"], alpha=0.25, color="tab:blue", label="raw")
    axes[0].plot(df["round"], df["score_rolling"], color="tab:blue",
                 label=f"rolling mean (window={rolling_window})")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Score per episode")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["round"], df["steps"], alpha=0.25, color="tab:orange", label="raw")
    axes[1].plot(df["round"], df["steps_rolling"], color="tab:orange",
                 label=f"rolling mean (window={rolling_window})")
    axes[1].set_xlabel("Episode (round)")
    axes[1].set_ylabel("Steps")
    axes[1].set_title("Steps per episode")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    print(f"\nSummary over {len(df)} episodes:")
    print(f"  Mean score:  {df['score'].mean():.3f}  (last {rolling_window}: {df['score_rolling'].iloc[-1]:.3f})")
    print(f"  Mean steps:  {df['steps'].mean():.1f}  (last {rolling_window}: {df['steps_rolling'].iloc[-1]:.1f})")

    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-rounds", type=int, default=200, help="Number of training episodes to run")
    parser.add_argument("--scenario", type=str, default="coin-heaven", help="Scenario to train on")
    parser.add_argument("--agent", type=str, default="agent_pgk", help="Agent directory name")
    parser.add_argument("--stats-dir", type=str, default=None,
                         help="Directory containing training_stats.csv. Defaults to "
                              "agent_code/<agent>, since that's where the agent's own "
                              "file writes (relative paths) actually land.")
    parser.add_argument("--rolling-window", type=int, default=20, help="Window size for the rolling mean")
    parser.add_argument("--out", type=str, default="learning_curve.png", help="Output plot filename")
    parser.add_argument("--skip-run", action="store_true", help="Skip training, just re-plot existing CSV")
    parser.add_argument("--fresh", action="store_true", help="Delete existing training_stats.csv before running")
    parser.add_argument("--show", action="store_true", help="Also open the plot in an interactive window")
    args = parser.parse_args()

    stats_dir = args.stats_dir or os.path.join("agent_code", args.agent)
    stats_path = os.path.join(stats_dir, STATS_FILENAME)

    if args.fresh:
        if os.path.isfile(stats_path):
            os.remove(stats_path)
            print(f"Removed old {stats_path}")

        model_path = os.path.join(stats_dir, "my-saved-model.pt")
        if os.path.isfile(model_path):
            os.remove(model_path)
            print(f"Removed old {model_path}")

    if not args.skip_run:
        run_training(args.n_rounds, args.scenario, args.agent)

    plot_stats(stats_path, args.rolling_window, args.out, args.show)


if __name__ == "__main__":
    main()