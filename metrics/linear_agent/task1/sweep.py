# sweep.py — run with: uv run metrics/linear_agent/task1/sweep.py
import os
import re
import shutil
import subprocess
import sys
from itertools import product
from pathlib import Path

# self-locating: metrics/linear_agent/task1/sweep.py -> repo root is 3 levels up
REPO_ROOT = Path(__file__).resolve().parents[3]
AGENT_DIR = REPO_ROOT / "agent_code" / "linear_agent"
RESULTS_DIR = REPO_ROOT / "results" / "linear_agent" / "task1"
N_ROUNDS = 2000
PYTHON = sys.executable

ALPHAS = [0.001, 0.01, 0.05, 0.1]
GAMMAS = [0.80, 0.90, 0.95, 0.99]

def patch(filepath, pattern, value):
    text = Path(filepath).read_text()
    text = re.sub(pattern, value, text, flags=re.MULTILINE)
    Path(filepath).write_text(text)

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    repo_root = REPO_ROOT

    # Save originals so we can restore them after the sweep
    train_orig    = (AGENT_DIR / "train.py").read_text()
    features_orig = (AGENT_DIR / "features.py").read_text()

    try:
        for alpha, gamma in product(ALPHAS, GAMMAS):
            print(f"\nRunning α={alpha}  γ={gamma} ...")

            # Patch hyperparameters in-place
            patch(AGENT_DIR / "train.py",
                  r"^ALPHA = [\d.]+", f"ALPHA = {alpha}")
            patch(AGENT_DIR / "features.py",
                  r"^GAMMA = [\d.]+", f"GAMMA = {gamma}")

            # Run the real game
            subprocess.run([
                PYTHON, "main.py", "play",
                "--no-gui",
                "--agents", "linear_agent",
                "--train", "1",
                "--scenario", "coin-heaven",
                "--n-rounds", str(N_ROUNDS),
            ], cwd=repo_root, check=True)

            # Save the CSV
            src = repo_root / "agent_code" / "linear_agent" / "metrics" / "training_log.csv"
            dst = RESULTS_DIR / f"log_a{alpha}_g{gamma}.csv"
            shutil.copy(src, dst)
            print(f"  Saved → {dst}")

    finally:
        # Always restore originals even if something crashes mid-sweep
        (AGENT_DIR / "train.py").write_text(train_orig)
        (AGENT_DIR / "features.py").write_text(features_orig)
        print("\nRestored original train.py and features.py")

if __name__ == "__main__":
    main()