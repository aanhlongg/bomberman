"""
Runs `my_agent` for N episodes in headless training mode, then plots the
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


def run_game(n_rounds: int, scenario: str, agent: str, eval_mode: bool, opponents: list, tag: str):
    cmd = [
        sys.executable, "main.py", "play",
        "--no-gui",
        "--agents", agent, *opponents,
        "--train", "1",
        "--scenario", scenario,
        "--n-rounds", str(n_rounds),
    ]
    env = os.environ.copy()
    if eval_mode:
        env["EVAL_MODE"] = "1"
    if tag:
        env["MODEL_TAG"] = tag
    print(f"Running: {' '.join(cmd)}"
          + (" [EVAL_MODE=1]" if eval_mode else "")
          + (f" [MODEL_TAG={tag}]" if tag else ""))
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"Warning: main.py exited with code {result.returncode}", file=sys.stderr)


def plot_stats(stats_path: str, rolling_window: int, out_path: str, show: bool, eval_mode: bool):
    if not os.path.isfile(stats_path):
        print(f"No {stats_path} found -- nothing to plot.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(stats_path)
    if df.empty:
        print(f"{stats_path} is empty -- nothing to plot.", file=sys.stderr)
        sys.exit(1)

    df["score_rolling"] = df["score"].rolling(rolling_window, min_periods=1).mean()
    df["steps_rolling"] = df["steps"].rolling(rolling_window, min_periods=1).mean()

    label = "test episode" if eval_mode else "episode (round)"
    subtitle = "Evaluation (fixed policy, no learning)" if eval_mode else "Training"

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(df["round"], df["score"], alpha=0.25, color="tab:blue", label="raw")
    axes[0].plot(df["round"], df["score_rolling"], color="tab:blue",
                 label=f"rolling mean (window={rolling_window})")
    axes[0].set_ylabel("Score")
    axes[0].set_title(f"Score per {label} -- {subtitle}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["round"], df["steps"], alpha=0.25, color="tab:orange", label="raw")
    axes[1].plot(df["round"], df["steps_rolling"], color="tab:orange",
                 label=f"rolling mean (window={rolling_window})")
    axes[1].set_xlabel(label.capitalize())
    axes[1].set_ylabel("Steps")
    axes[1].set_title(f"Steps per {label}")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    print(f"\nSummary over {len(df)} episodes:")
    print(f"  Mean score:  {df['score'].mean():.3f}  (std: {df['score'].std():.3f}, "
          f"min: {df['score'].min()}, max: {df['score'].max()})")
    print(f"  Mean steps:  {df['steps'].mean():.1f}  (std: {df['steps'].std():.1f})")

    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-rounds", type=int, default=200, help="Number of episodes to run")
    parser.add_argument("--scenario", type=str, default="coin-heaven", help="Scenario to run on")
    parser.add_argument("--agent", type=str, default="my_agent", help="Agent directory name")
    parser.add_argument("--opponents", type=str, nargs="*", default=[],
                         help="Additional agent names to play alongside your agent, e.g. --opponents rule_based_agent")
    parser.add_argument("--tag", type=str, default="",
                         help="Tag this training/eval run (e.g. 'solo', 'with_rulebased') so its "
                              "model and stats files don't collide with other runs of the same agent.")
    parser.add_argument("--eval", action="store_true",
                         help="Evaluate the already-trained model with a fixed (greedy) policy: "
                              "no exploration, no weight updates, no model file overwrite. "
                              "Logs to test_stats.csv instead of training_stats.csv.")
    parser.add_argument("--stats-dir", type=str, default=None,
                         help="Directory containing the stats CSV. Defaults to "
                              "agent_code/<agent>, since that's where the agent's own "
                              "file writes (relative paths) actually land.")
    parser.add_argument("--rolling-window", type=int, default=20, help="Window size for the rolling mean")
    parser.add_argument("--out", type=str, default=None, help="Output plot filename")
    parser.add_argument("--skip-run", action="store_true", help="Skip running the game, just re-plot existing CSV")
    parser.add_argument("--fresh", action="store_true", help="Delete existing stats CSV before running")
    parser.add_argument("--show", action="store_true", help="Also open the plot in an interactive window")
    args = parser.parse_args()

    stats_dir = args.stats_dir or os.path.join("agent_code", args.agent)
    kind = "test" if args.eval else "training"
    stats_filename = f"{kind}_stats{'_' + args.tag if args.tag else ''}.csv"
    stats_path = os.path.join(stats_dir, stats_filename)
    default_out = f"{'test' if args.eval else 'learning'}_curve{'_' + args.tag if args.tag else ''}.png"
    out_path = args.out or default_out

    if args.fresh and os.path.isfile(stats_path):
        os.remove(stats_path)
        print(f"Removed old {stats_path}")

    if not args.skip_run:
        run_game(args.n_rounds, args.scenario, args.agent, args.eval, args.opponents, args.tag)

    plot_stats(stats_path, args.rolling_window, out_path, args.show, args.eval)


if __name__ == "__main__":
    main()