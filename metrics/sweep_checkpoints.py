"""
evaluates every checkpoint of a training run and prints the best one.

Usage:
    uv run metrics/sweep_checkpoints.py --scenario loot-crate --jobs 8
    uv run metrics/sweep_checkpoints.py --opponents peaceful_agent coin_collector_agent --jobs 8
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS = os.path.join("agent_code", "linear_agent", "metrics", "checkpoints")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", default="linear_agent")
    parser.add_argument("--opponents", nargs="*", default=[])
    parser.add_argument("--scenario", default="classic")
    parser.add_argument("--n-rounds", type=int, default=60)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--jobs", type=int, default=6,
                        help="checkpoints to evaluate in parallel (default: 6)")
    args = parser.parse_args()

    directory = os.path.join(ROOT, CHECKPOINTS)
    rounds = sorted(
        int(name[len("weights_"):-len(".pt")])
        for name in os.listdir(directory)
        if name.startswith("weights_") and name.endswith(".pt")
    )

    def run(checkpoint):
        stats_path = tempfile.mktemp(suffix=".json")
        # separate log dir per job, parallel games writing the same logs corrupt them
        log_dir = tempfile.mkdtemp(prefix="sweep_logs_")
        env = dict(os.environ)
        env["LINEAR_AGENT_MODEL"] = os.path.join(directory, f"weights_{checkpoint}.pt")
        subprocess.run(
            [sys.executable, "main.py", "play", "--agents", args.agent, *args.opponents,
             "--scenario", args.scenario, "--no-gui", "--n-rounds", str(args.n_rounds),
             "--seed", str(args.seed), "--save-stats", stats_path, "--log-dir", log_dir],
            cwd=ROOT, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        stats = json.load(open(stats_path))
        os.remove(stats_path)
        mine = stats["by_agent"][args.agent]
        n = len(stats["by_round"])
        return {"checkpoint": checkpoint, "score": mine.get("score", 0) / n,
                "kills": mine.get("kills", 0), "coins": mine.get("coins", 0) / n,
                "suicides": mine.get("suicides", 0),
                "steps": mine.get("steps", 0) / n}

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(run, rounds))

    print(f"{'round':>6} {'score/rd':>9} {'kills':>7} {'coins/rd':>9} {'suicides':>9} {'steps/rd':>9}")
    for r in results:
        print(f"{r['checkpoint']:>6} {r['score']:>9.2f} {r['kills']:>7} "
              f"{r['coins']:>9.2f} {r['suicides']:>9} {r['steps']:>9.1f}")

    # on equal score prefer fewer steps, e.g. on coin-heaven every checkpoint gets all coins
    top = max(results, key=lambda r: (r["score"], -r["steps"]))
    best = (top["checkpoint"], top["score"])

    print(f"\nbest: round {best[0]} at {best[1]:.2f} score per round")
    print(f"install it with:\n"
          f"  cp {CHECKPOINTS}/weights_{best[0]}.pt "
          f"agent_code/{args.agent}/linear-model.pt")


if __name__ == "__main__":
    main()
