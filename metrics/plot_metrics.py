"""
plots training and evaluation metrics of a fully trained agent and different checkpoints.

Usage:
    uv run metrics/plot_metrics.py <training log> --agent <agent> --scenario <scenario> [--eval-rounds N] [--checkpoint-rounds N] [--title TITLE]

Example:
    uv run metrics/plot_metrics.py agent_code/linear_agent/metrics/training_log.csv --agent linear_agent --scenario loot-crate --eval-rounds 500 --checkpoint-rounds 25 --title "Linear agent - Task 2"
"""

import argparse
import csv
import json
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOOTHING = 100  # number of rounds for the moving average in the training curve


def load_training_log(csv_path):
    """
    returns each column of the training log as an array
    """
    with open(csv_path, newline="") as file:
        rows = list(csv.DictReader(file))

    return {name: np.array([float(row[name]) for row in rows]) for name in rows[0]}


def smooth(values):
    """
    average of the last SMOOTHING rounds (fewer at the start)
    """
    cumulative = np.concatenate([[0.0], np.cumsum(values)])
    ends = np.arange(1, len(values) + 1)
    starts = np.maximum(0, ends - SMOOTHING)
    return (cumulative[ends] - cumulative[starts]) / (ends - starts)


def evaluate(agent, scenario, n_rounds, stats_path, model_path=None, quiet=False):
    """
    plays n_rounds with a specified checkpoint of a model (or fully trained),
    and return the game's statistics per round and the agent's totals
    """
    env = dict(os.environ)
    if model_path:
        env["LINEAR_AGENT_MODEL"] = os.path.abspath(model_path)

    output = subprocess.DEVNULL if quiet else None
    subprocess.run(
        [
            sys.executable,
            "main.py",
            "play",
            "--agents",
            agent,
            "--scenario",
            scenario,
            "--no-gui",
            "--n-rounds",
            str(n_rounds),
            "--save-stats",
            stats_path,
        ],
        cwd=ROOT,
        env=env,
        stdout=output,
        stderr=output,
        check=True,
    )

    with open(stats_path) as file:
        stats = json.load(file)

    return list(stats["by_round"].values()), stats["by_agent"][agent]


def evaluate_checkpoints(directory, agent, scenario, n_rounds):
    """
    calls evaluate() for each checkpoint of a model
    """
    checkpoint_dir = os.path.join(directory, "checkpoints")
    if n_rounds <= 0 or not os.path.isdir(checkpoint_dir):
        return []

    snapshots = sorted(
        int(name[len("weights_") : -len(".pt")])
        for name in os.listdir(checkpoint_dir)
        if name.startswith("weights_") and name.endswith(".pt")
    )

    results = []
    for training_round in snapshots:
        base = os.path.join(checkpoint_dir, f"weights_{training_round}")
        rounds, _ = evaluate(
            agent,
            scenario,
            n_rounds,
            base + ".json",
            model_path=base + ".pt",
            quiet=True,
        )
        coins = sum(r["coins"] for r in rounds) / len(rounds)
        suicides = 100 * sum(r["suicides"] for r in rounds) / len(rounds)
        print(
            f"weights of round {training_round}: {coins:.1f} coins, {suicides:.0f}% suicides"
        )
        results.append((training_round, coins, suicides))

    return results


def evaluation_summary(rounds, totals, scenario):
    """
    print average value for each metric
    """
    n = len(rounds)
    coins = np.array([r["coins"] for r in rounds])
    steps = np.array([r["steps"] for r in rounds])
    suicides = sum(r["suicides"] for r in rounds)
    spread = coins.std(ddof=1) if n > 1 else 0.0

    return "\n".join(
        [
            f"evaluation: {n} greedy rounds on {scenario}",
            "",
            f"coins     {coins.mean():6.2f} ± {spread:.2f} (sd) per round",
            f"          min {coins.min()}, max {coins.max()}",
            f"steps     {steps.mean():6.1f} per round",
            f"suicides  {suicides} of {n} rounds ({100 * suicides / n:.1f}%)",
            f"invalid   {totals.get('invalid', 0) / n:6.2f} per round",
            f"bombs     {totals.get('bombs', 0) / n:6.2f} per round",
            f"crates    {totals.get('crates', 0) / n:6.2f} per round, "
            f"{totals.get('crates', 0) / max(totals.get('bombs', 0), 1):.2f} per bomb",
        ]
    )


def plot(
    log, checkpoints, checkpoint_games, rounds, totals, scenario, title, output_path
):
    episodes = log["round"]
    smoothed = f"average of {SMOOTHING} rounds"
    greedy = f"greedy agent ({checkpoint_games} games per snapshot)"

    fig = plt.figure(figsize=(12, 13))
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.9])
    fig.suptitle(f"{title}\n(training: exploring agent, evaluation: greedy agent)")

    coins = fig.add_subplot(grid[0, 0])
    coins.plot(
        episodes,
        log["coins"],
        color="lightgray",
        linewidth=0.5,
        label="exploring agent, per round",
    )
    coins.plot(episodes, smooth(log["coins"]), label=f"exploring agent, {smoothed}")
    if checkpoints:
        coins.plot(
            [c[0] for c in checkpoints],
            [c[1] for c in checkpoints],
            marker="o",
            label=greedy,
        )
    coins.set_title("Training: coins collected")
    coins.set_xlabel("training round")
    coins.set_ylabel("coins per round")
    coins.legend(fontsize=8)

    suicides = fig.add_subplot(grid[0, 1])
    suicides.plot(
        episodes, 100 * smooth(log["suicides"]), label=f"exploring agent, {smoothed}"
    )
    if checkpoints:
        suicides.plot(
            [c[0] for c in checkpoints],
            [c[2] for c in checkpoints],
            marker="o",
            label=greedy,
        )
    suicides.set_title("Training: suicide rate")
    suicides.set_xlabel("training round")
    suicides.set_ylabel("% of rounds")
    suicides.legend(fontsize=8)

    actions = fig.add_subplot(grid[1, 0])
    actions.plot(episodes, smooth(log["invalid_actions"]), label="invalid actions")
    actions.plot(episodes, smooth(log["bombs"]), label="bombs dropped")
    actions.set_title("Training: invalid actions and bombs")
    actions.set_xlabel("training round")
    actions.set_ylabel(f"per round ({smoothed})")
    actions.legend()

    weights = fig.add_subplot(grid[1, 1])
    for name in log:
        if name.startswith("w_"):
            weights.plot(episodes, log[name], label=name[2:], linewidth=1)
    weights.set_title("Training: learned weights")
    weights.set_xlabel("training round")
    weights.set_ylabel("weight")
    weights.legend(fontsize=7, ncol=2)

    evaluation_coins = np.array([r["coins"] for r in rounds])
    histogram = fig.add_subplot(grid[2, 0])
    histogram.hist(
        evaluation_coins,
        bins=np.arange(evaluation_coins.min(), evaluation_coins.max() + 2) - 0.5,
    )
    histogram.set_title("Evaluation: coins per round")
    histogram.set_xlabel("coins")
    histogram.set_ylabel("rounds")

    summary = fig.add_subplot(grid[2, 1])
    summary.axis("off")
    summary.text(
        0,
        1,
        evaluation_summary(rounds, totals, scenario),
        family="monospace",
        fontsize=11,
        va="top",
    )

    fig.tight_layout()
    fig.savefig(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "training_log",
        help="training log, e.g. agent_code/linear_agent/metrics/training_log.csv",
    )
    parser.add_argument(
        "--scenario", required=True, help="scenario to evaluate on, e.g. loot-crate"
    )
    parser.add_argument(
        "--eval-rounds",
        type=int,
        default=500,
        help="greedy rounds to evaluate the trained model (default: 500)",
    )
    parser.add_argument(
        "--checkpoint-rounds",
        type=int,
        default=25,
        help="greedy rounds per weight snapshot saved during training (default: 25, 0 skips them)",
    )
    parser.add_argument("--agent", required=True, help="agent to evaluate")
    parser.add_argument("--title", default=None, help="figure title")
    args = parser.parse_args()

    directory = os.path.dirname(os.path.abspath(args.training_log))
    output_path = os.path.join(directory, "metrics.png")
    stats_path = os.path.join(directory, "evaluation_stats.json")

    log = load_training_log(args.training_log)
    checkpoints = evaluate_checkpoints(
        directory, args.agent, args.scenario, args.checkpoint_rounds
    )
    rounds, totals = evaluate(args.agent, args.scenario, args.eval_rounds, stats_path)

    title = args.title or f"{args.agent} - {args.scenario}"
    plot(
        log,
        checkpoints,
        args.checkpoint_rounds,
        rounds,
        totals,
        args.scenario,
        title,
        output_path,
    )

    print(evaluation_summary(rounds, totals, args.scenario))
    print(f"\nsaved figure to {output_path}")
