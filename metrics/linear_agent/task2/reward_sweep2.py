# reward_sweep2.py — drop in repo root, run with: uv run python reward_sweep2.py
#
# Small, deliberate sweep over BOMB_DROPPED x COIN_FOUND, holding everything
# else fixed (WAITED=-0.15, CRATE_DESTROYED=1.5, COIN_COLLECTED=1.0,
# KILLED_SELF=-10.0, and all the structural fixes: doomed-bomb gate,
# crate_hit_potential, moves_to_crate + crate_potential). Each config trains
# for N_ROUNDS rounds on loot-crate, then evaluates pure-greedy for
# N_EVAL_ROUNDS rounds. Ranks by coins/round (primary) and steps/round
# (secondary, lower is better) per the user's stated goal: as many coins as
# possible in as few steps as possible.
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

AGENT_DIR = Path("agent_code/linear_agent")
RESULTS_DIR = Path("results/reward_sweep2")
N_ROUNDS = 5000
N_EVAL_ROUNDS = 200
PYTHON = sys.executable

# name -> (BOMB_DROPPED, COIN_FOUND)
CONFIGS = {
    "bomb-0.05_found1.0": (-0.05, 1.0),
    "bomb-0.05_found2.0": (-0.05, 2.0),
    "bomb0.0_found1.0": (0.0, 1.0),
    "bomb0.0_found2.0": (0.0, 2.0),
    "bomb0.2_found1.0": (0.2, 1.0),
    "bomb0.2_found2.0": (0.2, 2.0),
}


def patch_rewards(filepath, bomb_dropped, coin_found):
    text = Path(filepath).read_text()
    new_text, n1 = re.subn(r"(e\.BOMB_DROPPED:\s*)-?[\d.]+,", rf"\g<1>{bomb_dropped},", text)
    assert n1 == 1, f"expected 1 BOMB_DROPPED match, found {n1}"
    new_text, n2 = re.subn(r"(e\.COIN_FOUND:\s*)-?[\d.]+,", rf"\g<1>{coin_found},", new_text)
    assert n2 == 1, f"expected 1 COIN_FOUND match, found {n2}"
    Path(filepath).write_text(new_text)


def clean_agent_state():
    for f in ["linear-model.pt", "tracked_bomb_log.jsonl", "tracked_bomb_log.json"]:
        p = AGENT_DIR / f
        if p.exists():
            p.unlink()
    ckpt_dir = AGENT_DIR / "checkpoints"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)


def run(cmd, log_path):
    with open(log_path, "w") as logf:
        result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    return result.returncode


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).parent.resolve()
    train_path = AGENT_DIR / "train.py"
    train_orig = train_path.read_text()

    summary = {}

    try:
        for name, (bomb, found) in CONFIGS.items():
            print(f"\n=== Config '{name}': BOMB_DROPPED={bomb}, COIN_FOUND={found} ===", flush=True)
            patch_rewards(train_path, bomb, found)
            clean_agent_state()

            train_stats = RESULTS_DIR / f"train_{name}.json"
            train_log = RESULTS_DIR / f"train_{name}.log"
            rc = run(
                [
                    PYTHON, "main.py", "play",
                    "--agents", "linear_agent",
                    "--train", "1",
                    "--scenario", "loot-crate",
                    "--no-gui",
                    "--n-rounds", str(N_ROUNDS),
                    "--save-stats", str((repo_root / train_stats).resolve()),
                ],
                repo_root / train_log,
            )
            if rc != 0:
                print(f"  TRAINING FAILED (rc={rc}), see {train_log}")
                summary[name] = {"error": "training_failed"}
                continue

            # copy the trained model + weight vector aside for reference
            model_dst = RESULTS_DIR / f"model_{name}.pt"
            shutil.copy(AGENT_DIR / "linear-model.pt", model_dst)

            eval_stats = RESULTS_DIR / f"eval_{name}.json"
            eval_log = RESULTS_DIR / f"eval_{name}.log"
            rc = run(
                [
                    PYTHON, "main.py", "play",
                    "--agents", "linear_agent",
                    "--train", "0",
                    "--scenario", "loot-crate",
                    "--no-gui",
                    "--n-rounds", str(N_EVAL_ROUNDS),
                    "--save-stats", str((repo_root / eval_stats).resolve()),
                ],
                repo_root / eval_log,
            )
            if rc != 0:
                print(f"  EVAL FAILED (rc={rc}), see {eval_log}")
                summary[name] = {"error": "eval_failed"}
                continue

            with open(eval_stats) as f:
                d = json.load(f)
            by_round = d["by_round"]
            n = len(by_round)
            coins = sum(v["coins"] for v in by_round.values())
            suicides = sum(v["suicides"] for v in by_round.values())
            steps = sum(v["steps"] for v in by_round.values())

            result = {
                "bomb_dropped": bomb,
                "coin_found": found,
                "coins_per_round": coins / n,
                "steps_per_round": steps / n,
                "coins_per_step": coins / steps if steps else 0.0,
                "self_kill_rate": suicides / n,
                "total_coins": coins,
                "total_suicides": suicides,
            }
            summary[name] = result
            print(f"  -> coins/round={result['coins_per_round']:.2f} "
                  f"steps/round={result['steps_per_round']:.1f} "
                  f"self-kill={100*result['self_kill_rate']:.2f}%", flush=True)

    finally:
        train_path.write_text(train_orig)
        print("\nRestored original train.py")

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SWEEP SUMMARY ===")
    print(f"{'config':>22s} {'bomb':>6s} {'found':>6s} {'coins/rnd':>10s} {'steps/rnd':>10s} {'self-kill%':>11s}")
    for name, r in summary.items():
        if "error" in r:
            print(f"{name:>22s}  ERROR")
            continue
        print(f"{name:>22s} {r['bomb_dropped']:6.2f} {r['coin_found']:6.2f} "
              f"{r['coins_per_round']:10.2f} {r['steps_per_round']:10.1f} {100*r['self_kill_rate']:10.2f}%")


if __name__ == "__main__":
    main()
