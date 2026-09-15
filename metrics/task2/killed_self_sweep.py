# killed_self_sweep.py — drop in repo root, run with: python killed_self_sweep.py
#
# Controlled sweep over KILLED_SELF magnitude x seed, holding the current best
# configuration (masked-exploration only, no escape-survival bonus, no
# greedy-branch shielding) fixed everywhere else. Extends reward_sweep.py's
# patch/run/restore pattern with a second axis (seed): each of the 3
# KILLED_SELF values is run with the SAME 3 fixed seeds, so seed=N always
# means the same crate/coin layout and the same agent RNG stream regardless
# of which KILLED_SELF value is being tested. AGENT_SEED (read by
# callbacks.py's setup()) closes the gap that --seed alone only fixes the
# world's own RNG, not the agent's random exploration.
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

AGENT_DIR = Path("agent_code/linear_agent")
RESULTS_DIR = Path("results/killed_self_sweep")
N_ROUNDS = 1500
PYTHON = sys.executable

KILLED_SELF_VALUES = [-10.0, -5.0, -3.0]
SEEDS = [1, 2, 3]


def patch_killed_self(filepath, value):
    text = Path(filepath).read_text()
    new_text, n = re.subn(r"(KILLED_SELF_PENALTY = )-?[\d.]+", rf"\g<1>{value}", text)
    # check the pattern matched, not that the text changed -- the sweep's
    # first value (-10.0) equals the file's current default, so a no-op
    # substitution there is correct, not a sign the pattern wasn't found
    assert n == 1, f"expected exactly one KILLED_SELF_PENALTY match, found {n}"
    Path(filepath).write_text(new_text)


def tag_for(value):
    return f"k{abs(int(value))}"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).parent.resolve()
    train_path = AGENT_DIR / "train.py"
    train_orig = train_path.read_text()

    try:
        for value in KILLED_SELF_VALUES:
            patch_killed_self(train_path, value)

            for seed in SEEDS:
                run_tag = f"{tag_for(value)}_seed{seed}"
                print(f"\n=== KILLED_SELF={value}  seed={seed} ===")

                log_dir = RESULTS_DIR / f"logs_{run_tag}"
                log_dir.mkdir(parents=True, exist_ok=True)

                env = os.environ.copy()
                env["AGENT_SEED"] = str(seed)

                subprocess.run(
                    [
                        PYTHON, "main.py", "play",
                        "--no-gui",
                        "--agents", "linear_agent",
                        "--train", "1",
                        "--scenario", "loot-crate",
                        "--n-rounds", str(N_ROUNDS),
                        "--seed", str(seed),
                        "--log-dir", str(log_dir.resolve()),
                    ],
                    cwd=repo_root,
                    check=True,
                    env=env,
                )

                csv_src = repo_root / "agent_code" / "linear_agent" / "metrics" / "training_log.csv"
                csv_dst = RESULTS_DIR / f"log_{run_tag}.csv"
                shutil.copy(csv_src, csv_dst)

                model_src = repo_root / "agent_code" / "linear_agent" / "linear-model.pt"
                model_dst = RESULTS_DIR / f"model_{run_tag}.pt"
                shutil.copy(model_src, model_dst)

                print(f"  Saved -> {csv_dst}, {model_dst}, {log_dir / 'game.log'}")
    finally:
        train_path.write_text(train_orig)
        print("\nRestored original train.py")


if __name__ == "__main__":
    main()
