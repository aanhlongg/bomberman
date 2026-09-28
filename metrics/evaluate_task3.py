"""
evaluates a model against opponents (tasks 3 and 4) and prints score, kills,
coins and suicides. score is what the tournament counts: 1 per coin, 5 per kill.

Usage:
    uv run metrics/evaluate_task3.py --opponents peaceful_agent coin_collector_agent --n-rounds 200
    uv run metrics/evaluate_task3.py --opponents rule_based_agent --n-rounds 200
    uv run metrics/evaluate_task3.py --opponents rule_based_agent --model agent_code/linear_agent/metrics/checkpoints/weights_600.pt
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def evaluate(agent, opponents, scenario, n_rounds, model_path=None, seed=None):
    stats_path = tempfile.mktemp(suffix=".json")

    env = dict(os.environ)
    if model_path:
        env["LINEAR_AGENT_MODEL"] = env["TREE_AGENT_MODEL_PATH"] = os.path.abspath(model_path)

    command = [
        sys.executable, "main.py", "play",
        "--agents", agent, *opponents,
        "--scenario", scenario,
        "--no-gui",
        "--n-rounds", str(n_rounds),
        "--save-stats", stats_path,
    ]
    if seed is not None:
        command += ["--seed", str(seed)]

    subprocess.run(
        command, cwd=ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )

    with open(stats_path) as file:
        stats = json.load(file)
    os.remove(stats_path)
    return stats


def summarise(stats, agent, opponents, scenario, n_rounds):
    mine = stats["by_agent"][agent]

    coins = mine.get("coins", 0)
    kills = mine.get("kills", 0)
    suicides = mine.get("suicides", 0)
    score = mine.get("score", 0)
    steps = mine.get("steps", 0)

    lines = [
        f"vs {', '.join(opponents)}  ({n_rounds} rounds, scenario {scenario})",
        f"  score      {score:6d}   ({score/n_rounds:6.2f} per round)",
        f"  kills      {kills:6d}   ({100*kills/n_rounds:5.1f} per 100 rounds)",
        f"  coins      {coins:6d}   ({coins/n_rounds:6.2f} per round)",
        f"  suicides   {suicides:6d}   ({100*suicides/n_rounds:5.1f}%)",
        f"  steps      {steps/n_rounds:6.1f} per round",
    ]
    for opponent in opponents:
        other = stats["by_agent"].get(opponent, {})
        lines.append(
            f"  {opponent:<22} score {other.get('score',0):5d}  "
            f"kills {other.get('kills',0):3d}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", default="linear_agent")
    parser.add_argument("--opponents", nargs="+", required=True)
    parser.add_argument("--scenario", default="classic")
    parser.add_argument("--n-rounds", type=int, default=100)
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    stats = evaluate(
        args.agent, args.opponents, args.scenario,
        args.n_rounds, args.model, args.seed,
    )
    print(summarise(stats, args.agent, args.opponents, args.scenario, args.n_rounds))
