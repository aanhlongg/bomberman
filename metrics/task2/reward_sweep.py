# reward_sweep.py — drop in repo root, run with: python reward_sweep.py
#
# Small, deliberate reward-magnitude sweep for Task 2's bomb-safety features.
# Tests whether w_crate_hit / w_escape_exists turn positive when the BOMB_DROPPED
# cost is lowered and/or CRATE_DESTROYED / COIN_FOUND rewards are raised, relative
# to the current baseline values in train.py (BOMB_DROPPED=-2.0, CRATE_DESTROYED=0.75,
# COIN_FOUND=1.0). Patches train.py in place per config, runs a fresh 5000-round
# training session, saves the resulting metrics CSV, then restores train.py to its
# original content (even on failure), mirroring sweep.py's patch/run/restore pattern.
import re
import shutil
import subprocess
import sys
from pathlib import Path

AGENT_DIR = Path("agent_code/linear_agent")
RESULTS_DIR = Path("results/reward_sweep")
N_ROUNDS = 5000
PYTHON = sys.executable

# name -> (BOMB_DROPPED, CRATE_DESTROYED, COIN_FOUND); KILLED_SELF stays fixed at -10.0
CONFIGS = {
    "cheap_bombing": (-0.5, 0.75, 1.0),
    "rich_rewards": (-2.0, 2.0, 2.0),
    "both": (-0.5, 2.0, 2.0),
}


def patch_rewards(filepath, bomb_dropped, crate_destroyed, coin_found):
    text = Path(filepath).read_text()
    text = re.sub(r"(e\.BOMB_DROPPED:\s*)-?[\d.]+,", rf"\g<1>{bomb_dropped},", text)
    text = re.sub(r"(e\.CRATE_DESTROYED:\s*)-?[\d.]+,", rf"\g<1>{crate_destroyed},", text)
    text = re.sub(r"(e\.COIN_FOUND:\s*)-?[\d.]+,", rf"\g<1>{coin_found},", text)
    Path(filepath).write_text(text)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).parent.resolve()
    train_path = AGENT_DIR / "train.py"
    train_orig = train_path.read_text()

    try:
        for name, (bomb, crate, coin) in CONFIGS.items():
            print(f"\n=== Running '{name}': BOMB_DROPPED={bomb}, CRATE_DESTROYED={crate}, COIN_FOUND={coin} ===")
            patch_rewards(train_path, bomb, crate, coin)

            # each config's game.log goes to its own directory, so per-round
            # survival/self-kill events aren't overwritten by the next config's run
            log_dir = RESULTS_DIR / f"logs_{name}"
            log_dir.mkdir(parents=True, exist_ok=True)

            subprocess.run(
                [
                    PYTHON, "main.py", "play",
                    "--no-gui",
                    "--agents", "linear_agent",
                    "--train", "1",
                    "--scenario", "loot-crate",
                    "--n-rounds", str(N_ROUNDS),
                    "--log-dir", str(log_dir.resolve()),
                ],
                cwd=repo_root,
                check=True,
            )

            src = repo_root / "agent_code" / "linear_agent" / "metrics" / "training_log.csv"
            dst = RESULTS_DIR / f"log_{name}.csv"
            shutil.copy(src, dst)
            print(f"  Saved -> {dst} (game log -> {log_dir / 'game.log'})")
    finally:
        train_path.write_text(train_orig)
        print("\nRestored original train.py")


if __name__ == "__main__":
    main()
