"""
turns the following metrics into a 2x2 plot:

- sparse/shaped reward
- steps needed to collect all coins
- invalid action count
- weight for "moves_to_coin" feature

Usage:
    uv run python <path/to/logs.csv> --title "<title>"

Example:
    uv run python metrics/plot_metrics.py agent_code/linear_agent/metrics/training_log.csv --title "Linear agent - Task 1 - Training"
    uv run python metrics/plot_metrics.py agent_code/linear_agent/metrics/evaluation_log.csv --title "Linear agent - Task 1 - Evaluation"
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def load_episodes(csv_path):
    episodes = defaultdict(
        lambda: {
            "sparse": 0.0,
            "shaped": 0.0,
            "invalid": 0,
            "steps": 0,
            "w_moves_to_coin": 0.0,
        }
    )

    with open(csv_path, newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            episode = episodes[int(row["round"])]
            episode["sparse"] += float(row["sparse_reward"])
            episode["shaped"] += float(row["shaped_reward"])
            episode["invalid"] += 0 if int(row["valid"]) else 1
            episode["steps"] = max(episode["steps"], int(row["step"]))
            episode["w_moves_to_coin"] = float(row["w_moves_to_coin"])

    return episodes


def plot(csv_path, title, output_path):
    episodes = load_episodes(csv_path)
    rounds = sorted(episodes)

    sparse = [episodes[r]["sparse"] for r in rounds]
    shaped = [episodes[r]["shaped"] for r in rounds]
    steps = [episodes[r]["steps"] for r in rounds]
    invalid = [episodes[r]["invalid"] for r in rounds]
    w_moves_to_coin = [episodes[r]["w_moves_to_coin"] for r in rounds]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(title)

    axes[0, 0].plot(rounds, sparse, label="sparse reward")
    axes[0, 0].plot(rounds, shaped, label="shaped reward")
    axes[0, 0].set_title("Reward per episode")
    axes[0, 0].set_xlabel("episode")
    axes[0, 0].set_ylabel("reward")
    axes[0, 0].legend()

    axes[0, 1].plot(rounds, steps)
    axes[0, 1].set_title("Steps per episode")
    axes[0, 1].set_xlabel("episode")
    axes[0, 1].set_ylabel("steps")

    axes[1, 0].plot(rounds, invalid)
    axes[1, 0].set_title("Invalid actions per episode")
    axes[1, 0].set_xlabel("episode")
    axes[1, 0].set_ylabel("invalid actions")

    axes[1, 1].plot(rounds, w_moves_to_coin)
    axes[1, 1].set_title("Learned weight: moves toward coin")
    axes[1, 1].set_xlabel("episode")
    axes[1, 1].set_ylabel("w[moves_to_coin]")

    fig.tight_layout()
    fig.savefig(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        help="path to a metrics CSV, e.g. agent_code/linear_agent/metrics/train_log.csv",
    )
    parser.add_argument(
        "--title", default=None, help="figure title (defaults to the csv path)"
    )
    parser.add_argument(
        "--output", default=None, help="output image path (defaults next to the csv)"
    )
    args = parser.parse_args()

    output_path = args.output or os.path.splitext(args.csv_path)[0] + ".png"
    plot(args.csv_path, args.title or args.csv_path, output_path)
    print(f"saved figure to {output_path}")
