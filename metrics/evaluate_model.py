"""
evaluates one model over n greedy rounds and prints the average stats.

Usage:
    uv run metrics/evaluate_model.py --agent <agent> --scenario <scenario> --n-rounds N [--model PATH] [--seed N]

Example:
    uv run metrics/evaluate_model.py --agent linear_agent --scenario loot-crate --n-rounds 300
    uv run metrics/evaluate_model.py --agent linear_agent --scenario loot-crate --n-rounds 300 --model agent_code/linear_agent/metrics/checkpoints/weights_600.pt
"""

import argparse
import os
import tempfile

from plot_metrics import evaluate, evaluation_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", required=True, help="agent to evaluate")
    parser.add_argument("--scenario", required=True, help="scenario to evaluate on, e.g. loot-crate")
    parser.add_argument("--n-rounds", type=int, default=300, help="greedy rounds to play (default: 300)")
    parser.add_argument("--model", default=None, help="LINEAR_AGENT_MODEL override (default: the agent's own linear-model.pt)")
    args = parser.parse_args()

    stats_path = tempfile.mktemp(suffix=".json")
    rounds, totals = evaluate(
        args.agent, args.scenario, args.n_rounds, stats_path, model_path=args.model, quiet=True
    )
    os.remove(stats_path)

    print(evaluation_summary(rounds, totals, args.scenario))
