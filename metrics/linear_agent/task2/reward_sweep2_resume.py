# reward_sweep2_resume.py — completes the remaining configs from the
# reward_sweep2.py run that was killed after 3/6 configs finished.
# train.py is currently already patched to bomb0.0_found2.0 (the 4th
# config, interrupted mid-run) -- reuse that as-is instead of re-patching,
# then continue to the last two configs. At the end, restores train.py to
# whichever config's summary.json (loaded from the first 3 completed runs
# plus these 3) turns out to win on coins/round, so no separate "apply the
# winner" step is needed afterward.
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

REMAINING = {
    "bomb0.0_found2.0": (0.0, 2.0),   # already patched on disk, just run it
    "bomb0.2_found1.0": (0.2, 1.0),
    "bomb0.2_found2.0": (0.2, 2.0),
}
ALREADY_DONE = ["bomb-0.05_found1.0", "bomb-0.05_found2.0", "bomb0.0_found1.0"]


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


def eval_result(name, bomb, found, repo_root):
    clean_agent_state()
    train_stats = RESULTS_DIR / f"train_{name}.json"
    train_log = RESULTS_DIR / f"train_{name}.log"
    rc = run(
        [PYTHON, "main.py", "play", "--agents", "linear_agent", "--train", "1",
         "--scenario", "loot-crate", "--no-gui", "--n-rounds", str(N_ROUNDS),
         "--save-stats", str((repo_root / train_stats).resolve())],
        repo_root / train_log,
    )
    if rc != 0:
        return {"error": "training_failed"}

    shutil.copy(AGENT_DIR / "linear-model.pt", RESULTS_DIR / f"model_{name}.pt")

    eval_stats = RESULTS_DIR / f"eval_{name}.json"
    eval_log = RESULTS_DIR / f"eval_{name}.log"
    rc = run(
        [PYTHON, "main.py", "play", "--agents", "linear_agent", "--train", "0",
         "--scenario", "loot-crate", "--no-gui", "--n-rounds", str(N_EVAL_ROUNDS),
         "--save-stats", str((repo_root / eval_stats).resolve())],
        repo_root / eval_log,
    )
    if rc != 0:
        return {"error": "eval_failed"}

    with open(eval_stats) as f:
        d = json.load(f)
    by_round = d["by_round"]
    n = len(by_round)
    coins = sum(v["coins"] for v in by_round.values())
    suicides = sum(v["suicides"] for v in by_round.values())
    steps = sum(v["steps"] for v in by_round.values())
    return {
        "bomb_dropped": bomb, "coin_found": found,
        "coins_per_round": coins / n, "steps_per_round": steps / n,
        "coins_per_step": coins / steps if steps else 0.0,
        "self_kill_rate": suicides / n,
        "total_coins": coins, "total_suicides": suicides,
    }


def load_existing(name):
    with open(RESULTS_DIR / f"eval_{name}.json") as f:
        d = json.load(f)
    by_round = d["by_round"]
    n = len(by_round)
    coins = sum(v["coins"] for v in by_round.values())
    suicides = sum(v["suicides"] for v in by_round.values())
    steps = sum(v["steps"] for v in by_round.values())
    # bomb/found values recovered from the config name convention
    return {
        "coins_per_round": coins / n, "steps_per_round": steps / n,
        "coins_per_step": coins / steps if steps else 0.0,
        "self_kill_rate": suicides / n,
        "total_coins": coins, "total_suicides": suicides,
    }


def main():
    repo_root = Path(__file__).parent.resolve()
    train_path = AGENT_DIR / "train.py"

    summary = {}
    for name in ALREADY_DONE:
        summary[name] = load_existing(name)
        print(f"(loaded existing) {name}: coins/round={summary[name]['coins_per_round']:.2f}")

    first = True
    for name, (bomb, found) in REMAINING.items():
        print(f"\n=== Config '{name}': BOMB_DROPPED={bomb}, COIN_FOUND={found} ===", flush=True)
        if not first:
            patch_rewards(train_path, bomb, found)
        first = False
        result = eval_result(name, bomb, found, repo_root)
        summary[name] = result
        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  -> coins/round={result['coins_per_round']:.2f} "
                  f"steps/round={result['steps_per_round']:.1f} "
                  f"self-kill={100*result['self_kill_rate']:.2f}%", flush=True)

    # attach bomb/found values to the loaded-existing entries too, parsed from name
    name_to_params = {
        "bomb-0.05_found1.0": (-0.05, 1.0), "bomb-0.05_found2.0": (-0.05, 2.0),
        "bomb0.0_found1.0": (0.0, 1.0), "bomb0.0_found2.0": (0.0, 2.0),
        "bomb0.2_found1.0": (0.2, 1.0), "bomb0.2_found2.0": (0.2, 2.0),
    }
    for name, (b, f_) in name_to_params.items():
        if "error" not in summary[name]:
            summary[name]["bomb_dropped"] = b
            summary[name]["coin_found"] = f_

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== FULL SWEEP SUMMARY ===")
    print(f"{'config':>22s} {'bomb':>6s} {'found':>6s} {'coins/rnd':>10s} {'steps/rnd':>10s} {'self-kill%':>11s}")
    valid = {n: r for n, r in summary.items() if "error" not in r}
    for name, r in summary.items():
        if "error" in r:
            print(f"{name:>22s}  ERROR")
            continue
        print(f"{name:>22s} {r['bomb_dropped']:6.2f} {r['coin_found']:6.2f} "
              f"{r['coins_per_round']:10.2f} {r['steps_per_round']:10.1f} {100*r['self_kill_rate']:10.2f}%")

    if valid:
        winner = max(valid.items(), key=lambda kv: kv[1]["coins_per_round"])
        wname, wres = winner
        print(f"\nWinner (by coins/round): {wname} (BOMB_DROPPED={wres['bomb_dropped']}, COIN_FOUND={wres['coin_found']})")
        patch_rewards(train_path, wres["bomb_dropped"], wres["coin_found"])
        print("train.py patched to the winning config.")


if __name__ == "__main__":
    main()
