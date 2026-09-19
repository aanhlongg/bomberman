"""
refit_interval_sweep.py -- re-sweeping REFIT_INTERVAL against Task 2's
workload, not Task 1's
=============================================================================
Run from anywhere after cloning (self-locating, resolves the repo root from
its own file path):

    uv run metrics/tree_agent/refit_interval_sweep.py

Motivation: every tree_agent task since Task 1 has reused Task 1's
sweep-selected REFIT_INTERVAL=25 as-is (train.py's own comment admits this
was never re-swept for Task 2+'s own round-length distribution). Task 1's
coin-heaven rounds end the instant all coins are grabbed (observed ~124
steps/round for the final model); Task 2-4's crate-clearing rounds run much
closer to the full 400-step cap, so the replay buffer fills at a very
different rate and REFIT_INTERVAL=25 may no longer be well-matched to how
much new experience accumulates between refits.

Only REFIT_INTERVAL is swept here, not N_ESTIMATORS: Task 1's own sweep
found N_ESTIMATORS had no measurable effect on final quality across its
whole tested range (every cell converged to the same ceiling), so
re-testing it against a different scenario is a low-expected-value use of
sweep time. N_ESTIMATORS is left at train.py's shipped default (50).

Each config trains a fresh model for N_ROUNDS on loot-crate (Task 2's own
scenario -- the shared workload Task 3/4 both build on) and its
--save-stats by_round breakdown is used directly to read off final
coins/crates/bombs/suicides, rather than a separate pure-greedy evaluation
pass, to keep sweep cost down (mirrors Task 1's own sweep methodology of
reading training-time metrics directly rather than a dedicated eval per
config).

self.model is always reset to None at the start of any --train run (see
callbacks.setup()), so no explicit reset between sweep configs is needed.
tree-model.pt itself, and train.py's REFIT_INTERVAL line, are still
saved/restored around the sweep purely so a crash or interruption doesn't
leave the working tree_agent in a modified state.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = REPO_ROOT / "agent_code" / "tree_agent"
RESULTS_DIR = REPO_ROOT / "results" / "tree_agent" / "refit_interval_sweep"
N_ROUNDS = 2000
PYTHON = sys.executable

REFIT_INTERVALS = [10, 25, 50, 100]


def patch(filepath: Path, pattern: str, value: str) -> None:
    text = filepath.read_text()
    text = re.sub(pattern, value, text, flags=re.MULTILINE)
    filepath.write_text(text)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_py = AGENT_DIR / "train.py"
    model_path = AGENT_DIR / "tree-model.pt"

    train_orig = train_py.read_text()
    model_orig = model_path.read_bytes() if model_path.is_file() else None

    assert re.search(r"^REFIT_INTERVAL = \d+", train_orig, flags=re.MULTILINE), \
        "REFIT_INTERVAL pattern didn't match -- aborting before silently no-op patching"

    try:
        for refit_interval in REFIT_INTERVALS:
            print(f"\nRunning REFIT_INTERVAL={refit_interval} ({N_ROUNDS} rounds, loot-crate) ...")

            before = train_py.read_text()
            patch(train_py, r"^REFIT_INTERVAL = \d+", f"REFIT_INTERVAL = {refit_interval}")
            after = train_py.read_text()
            assert before != after, f"Patch for REFIT_INTERVAL={refit_interval} was a no-op"

            out_json = RESULTS_DIR / f"refit_{refit_interval}_train.json"
            subprocess.run(
                [
                    PYTHON, "main.py", "play",
                    "--no-gui",
                    "--agents", "tree_agent",
                    "--train", "1",
                    "--scenario", "loot-crate",
                    "--n-rounds", str(N_ROUNDS),
                    "--save-stats", str(out_json),
                ],
                cwd=REPO_ROOT,
                check=True,
            )
            print(f"  Saved -> {out_json}")

    finally:
        train_py.write_text(train_orig)
        if model_orig is not None:
            model_path.write_bytes(model_orig)
        elif model_path.is_file():
            model_path.unlink()
        print("\nRestored original train.py and tree-model.pt")


if __name__ == "__main__":
    main()
