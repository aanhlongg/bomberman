"""
sweep.py -- Fitted-Q-Iteration hyperparameter sweep for tree_agent's Task 1
=============================================================================
Run from anywhere after cloning (self-locating, resolves the repo root from
its own file path):

    uv run metrics/tree_agent/task1/sweep.py

Mirrors metrics/linear_agent/task1/sweep.py's methodology: vary exactly two
hyperparameters over a small grid (holding everything else fixed), training
one fresh model per combination on coin-heaven, and archiving each run's
per-step training_log.csv for later plotting.

REFIT_INTERVAL (how often the replay buffer gets batch-refit into a fresh
ExtraTreesRegressor) and N_ESTIMATORS (ensemble size) were chosen as the
two swept parameters: they're the pair most likely to trade off convergence
quality against wall-clock cost, mirroring the role alpha/gamma played for
linear_agent's semi-gradient updates. MIN_SAMPLES_LEAF and FQI_SWEEPS are
left at train.py's current defaults (5 and 3) throughout.

self.model is always reset to None at the start of any --train run (see
callbacks.setup()), so no explicit reset between sweep configs is needed --
each one trains a genuinely fresh model regardless of what tree-model.pt
currently holds. tree-model.pt itself, and train.py's hyperparameter lines,
are still saved/restored around the sweep purely so a crash or interruption
doesn't leave the working tree_agent in a modified state.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
AGENT_DIR = REPO_ROOT / "agent_code" / "tree_agent"
RESULTS_DIR = REPO_ROOT / "results" / "tree_agent" / "task1"
N_ROUNDS = 2000
PYTHON = sys.executable

REFIT_INTERVALS = [25, 50, 100, 200]
N_ESTIMATORS_GRID = [20, 50, 100, 150]


def patch(filepath: Path, pattern: str, value: str) -> None:
    text = filepath.read_text()
    text = re.sub(pattern, value, text, flags=re.MULTILINE)
    filepath.write_text(text)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_py = AGENT_DIR / "train.py"
    model_path = AGENT_DIR / "tree-model.pt"
    log_path = AGENT_DIR / "metrics" / "training_log.csv"

    train_orig = train_py.read_text()
    model_orig = model_path.read_bytes() if model_path.is_file() else None

    try:
        for refit_interval, n_estimators in product(REFIT_INTERVALS, N_ESTIMATORS_GRID):
            print(f"\nRunning REFIT_INTERVAL={refit_interval}  N_ESTIMATORS={n_estimators} ...")

            patch(train_py, r"^REFIT_INTERVAL = \d+", f"REFIT_INTERVAL = {refit_interval}")
            patch(train_py, r"^N_ESTIMATORS = \d+", f"N_ESTIMATORS = {n_estimators}")

            subprocess.run(
                [
                    PYTHON, "main.py", "play",
                    "--no-gui",
                    "--agents", "tree_agent",
                    "--train", "1",
                    "--scenario", "coin-heaven",
                    "--n-rounds", str(N_ROUNDS),
                ],
                cwd=REPO_ROOT,
                check=True,
            )

            dst = RESULTS_DIR / f"log_ri{refit_interval}_ne{n_estimators}.csv"
            shutil.copy(log_path, dst)
            print(f"  Saved -> {dst}")

    finally:
        train_py.write_text(train_orig)
        if model_orig is not None:
            model_path.write_bytes(model_orig)
        elif model_path.is_file():
            model_path.unlink()
        print("\nRestored original train.py and tree-model.pt")


if __name__ == "__main__":
    main()
