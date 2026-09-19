<div align="center">
    <h1> Reinforcement Learning Agents in Bomberman </h1>
    <h3></h3>
</div>

Two agents, built on deliberately different function-approximation architectures (per the
assignment's requirement to develop two or more different models): **`linear_agent`**
(`Q(s,a) = weights . features(s,a)`, trained online via semi-gradient TD(0)) and **`tree_agent`**
(`Q(s,a) = model.predict(features(s,a))`, a tree ensemble trained via Fitted Q-Iteration). Both
are developed task-by-task, following the assignment's four subgoals; each section below
documents one task for one agent, with the commands to reproduce it and the validated result.

## linear_agent

Represents `Q(s,a)` as a dot product between a weight vector and a small set of hand-crafted,
interpretable state-action features. No neural network, replay buffer, or target network --
`self.weights` (see `callbacks.py`) is the entire model. Complete through all four tasks; full
narrative detail (every fix attempted, including reverted ones) lives in
`report/report1_body.typ` through `report/report4_body.typ`.

### Task 1: collecting coins

train the linear agent:
```bash
uv run main.py play --agents linear_agent --train 1 --scenario coin-heaven --no-gui --n-rounds 1000
```

evaluate the linear agent:
```bash
uv run main.py play --agents linear_agent --scenario coin-heaven # single run 
uv run main.py play --agents linear_agent --scenario coin-heaven --no-gui --n-rounds 200 # multiple runs
```

### Task 2: clearing crates

The agent lives entirely in three files: `agent_code/linear_agent/features.py`, `train.py`,
`callbacks.py`. To reproduce it in a fresh checkout, drop these three files into
`agent_code/linear_agent/` of this repository (overwriting the Task 1 versions) -- no other
files are needed, and no pretrained model needs to be shipped alongside them.

train the linear agent (starts from scratch every time `--train 1` is passed, regardless of any
existing `linear-model.pt`):
```bash
uv run main.py play --agents linear_agent --train 1 --scenario loot-crate --no-gui --n-rounds 50000
```

**50,000 rounds is what the reported result below was trained and validated on** (commit-gap
and escape-safety checked stable across three independent runs at 5,000 / 20,000 / 50,000
rounds). Training takes roughly 1-4 hours depending on machine and how large the per-run log
files grow (see note below); 20,000 rounds already gets close and is a reasonable choice if
short on time.

evaluate the linear agent (loads `linear-model.pt`, no exploration, no further training):
```bash
uv run main.py play --agents linear_agent --scenario loot-crate # single run
uv run main.py play --agents linear_agent --scenario loot-crate --no-gui --n-rounds 200 --save-stats results.json # multiple runs
```

Expected result on `loot-crate` (`CRATE_DENSITY = 0.75`, 50 coins hidden under crates, 400 steps
per round), averaged over a 200-round evaluation:

| Metric | Value |
|---|---|
| Coins/round | ~46.7 / 50 (93%) |
| Self-kill rate | 0% |
| Full-board-clear rate | ~16% of rounds |

**Note on train vs. eval behavior:** the agent's escape-routing (which safe tile it retreats to
after dropping a bomb, when multiple equally-short options exist) is intentionally different
during training vs. evaluation -- training always uses the simpler routing the saved weights
were originally validated under, while evaluation uses an improved, target-aware routing that
noticeably shortens the average time to clear a board. This is automatic (gated on `self.train`
in `features.py`'s `state_features`) and requires no action from you; it's mentioned here only
so the difference isn't mistaken for a bug if you go looking at the code.

**Note on training speed:** two files grow large over a long run
(`agent_code/linear_agent/metrics/training_log.csv` and `agent_code/linear_agent/logs/linear_agent.log`,
both per-step logs) and have been observed to measurably slow training down past ~20,000 rounds
on some machines. If a run seems to be slowing down significantly, deleting those two files
mid-run is safe (the process keeps writing to them regardless; only their on-disk footprint
changes) and has been observed to restore normal speed.

### Task 3: hunting opponents

train against either opponent (5,000 rounds is what the shipped model was trained/validated on):
```bash
uv run main.py play --agents linear_agent peaceful_agent --train 1 --scenario classic --no-gui --n-rounds 5000
uv run main.py play --agents linear_agent coin_collector_agent --train 1 --scenario classic --no-gui --n-rounds 5000
```

evaluate:
```bash
uv run main.py play --agents linear_agent peaceful_agent --scenario classic --no-gui --n-rounds 200 --save-stats results.json
uv run main.py play --agents linear_agent coin_collector_agent --scenario classic --no-gui --n-rounds 200 --save-stats results.json
```

Expected result of the currently shipped model (`linear-model.pt`, trained against `rule_based_agent`
-- see Task 4 below; it also handles Task 3's opponents well as an emergent side effect, see
`report3_body.typ` / `report4_body.typ`), averaged over 200 evaluation rounds:

| Opponent | Score (ours / theirs) | Our kills | Our suicides |
|---|---|---|---|
| `peaceful_agent` | 2550 / 4 | 154 (77%) | 5 (2.5%) |
| `coin_collector_agent` | 1299 / 654 | 29 (14.5%) | 31 (15.5%) |

`coin_collector_agent`'s stalemate rate (rounds that end with the board fully cleared but neither
agent finishing the other off) is a known, documented limitation -- see the "Open Questions"
section of `report3_body.typ` for the full investigation and every fix attempted.

### Task 4: holding your own against rule_based_agent

train:
```bash
uv run main.py play --agents linear_agent rule_based_agent --train 1 --scenario classic --no-gui --n-rounds 5000
```

evaluate (1-vs-1, and the 4-player free-for-all used to check generalization):
```bash
uv run main.py play --agents linear_agent rule_based_agent --scenario classic --no-gui --n-rounds 200 --save-stats results.json
uv run main.py play --agents linear_agent rule_based_agent rule_based_agent rule_based_agent --scenario classic --no-gui --n-rounds 200 --save-stats results.json
```

Expected result of the shipped model, averaged over 200 evaluation rounds:

| Matchup | Score (ours / theirs) | Our kills | Our suicides |
|---|---|---|---|
| 1v1 vs `rule_based_agent` | 1190 / 825 | 23 (11.5%) | 41 (20.5%) |
| 4-player FFA (ours vs 3x `rule_based_agent`) | 826 vs ~519 avg | 29 (14.5%) | 46 (23%, vs. ~42.5% avg for the three `rule_based_agent` copies) |

The 1v1 result is validated across an independent 5-seed training sweep (suicides 17-22%,
score 1220-1280 vs. 712-794 across all 5 seeds), not just this single run -- see
`report4_body.typ` for the full sweep table and every fix attempted (including several that
were tried and reverted after regressing other opponents).

## tree_agent

Represents `Q(s,a)` as the output of a tree-ensemble regressor (`ExtraTreesRegressor`), trained
via Fitted Q-Iteration (Ernst, Geurts & Wehenkel, 2005) -- a batch method that periodically
refits on an accumulated replay buffer rather than updating incrementally, so there is no
per-step SGD update and no equivalent of `linear_agent`'s synthetic counterfactual weight-nudges.
Danger detection, escape routing, and hunting logic are ported from `linear_agent`'s own
(architecture-independent) implementation almost verbatim, since those are properties of the
game state, not the function approximator; what does and doesn't transfer between the two agents
is documented per-task in `report/report_tree_taskN_body.typ`. Like `linear_agent`, the agent
lives entirely in three files (`agent_code/tree_agent/features.py`, `train.py`, `callbacks.py`)
and needs no pretrained model shipped alongside them to reproduce from scratch
(`agent_code/tree_agent/tests/` holds optional unit tests, see the Metrics section).

Unlike `linear_agent`, the current version also carries a deterministic **navigation-target
selector** in `features.py` (`_select_target`): each step, every visible coin and every tile a
bomb could clear at least one crate from (and be escaped from) gets one score, and the best one
becomes *the* target the learned Q-function's distance features and shaping potentials refer to.
The model still chooses every action (`argmax_a Q(s,a)` over the safety-filtered actions, 16
features per state-action pair); the selector only defines what "toward the target" means -- see
the `_select_target` docstring and the `COIN_TARGET_VALUE` comment in `features.py`.

### Task 1: collecting coins

train (2,000 rounds, the sweep-selected config `REFIT_INTERVAL = 25` -- see
`metrics/tree_agent/task1/sweep.py` for the 16-configuration hyperparameter sweep behind that
choice):
```bash
uv run main.py play --agents tree_agent --train 1 --scenario coin-heaven --no-gui --n-rounds 2000
```

evaluate (loads `tree-model.pt`, no exploration, no further training):
```bash
uv run main.py play --agents tree_agent --scenario coin-heaven # single run
uv run main.py play --agents tree_agent --scenario coin-heaven --no-gui --n-rounds 200 --save-stats results.json # multiple runs
```

Expected result on `coin-heaven`, averaged over a 200-round evaluation:

| Metric | Value |
|---|---|
| Coins/round | 50.0 / 50 (100%) |
| Invalid actions | 0 across all 200 rounds |
| Avg. steps/round | 124.1 |

This exceeds `linear_agent`'s own Task 1 result (48.2/50, 96.4%) -- see
`report/report_tree_task1_body.typ` for the full sweep and validation writeup. The table was
measured with the Task 1 model of the time; the currently shipped `tree-model.pt` (the Task 2
routing model, see below) reproduces it without retraining: 50.0/50, 0 invalid actions, 126.9
steps/round over a 50-round evaluation on `coin-heaven`.

### Task 2: clearing crates

train (1,500 rounds is what the shipped model was trained on; a 5,000-round run of the same code
evaluated slightly *worse* -- 49.57 coins/round, 187/200 full clears, one 0-coin round -- so the
shorter run is the one shipped):
```bash
uv run main.py play --agents tree_agent --train 1 --scenario loot-crate --no-gui --n-rounds 1500
```

evaluate (loads `tree-model.pt`, no exploration, no further training):
```bash
uv run main.py play --agents tree_agent --scenario loot-crate # single run
uv run main.py play --agents tree_agent --scenario loot-crate --no-gui --n-rounds 200 --save-stats results.json # multiple runs
```

Expected result on `loot-crate` (`CRATE_DENSITY = 0.75`, 50 coins hidden under crates, 400 steps
per round), averaged over a 200-round pure-greedy evaluation of the shipped model:

| Metric | Value |
|---|---|
| Coins/round | 49.95 / 50 (99.9%), no round below 46 |
| Full-board clears (all 50 coins) | 197 / 200 (98.5%) |
| Round length | median 352 steps (min 305) -- rounds end early once the board is clear |
| Bombs/round | 38.4 (3.19 crates per bomb) |
| Crates cleared/round | 122.6 |
| Self-kill rate | 0% |
| Invalid actions | 0 across all 200 rounds |
| Steps per bomb cycle | 8.9 (median 7, the physical minimum) |

This beats `linear_agent`'s Task 2 result (46.7/50, ~16% full clears) and the previous
`tree_agent` version (44.90/50, 0% full clears, 11.7 steps per bomb cycle).

**Model compatibility:** `callbacks.setup` checks the loaded model's feature count against
`N_FEATURES` and raises a `ValueError` on a mismatch instead of playing with a mismatched
model. `agent_code/tree_agent/tree-model.pt` is the 16-feature model above; the archived
`results/tree_agent/**/tree_model_*.pt` files are 11-feature models from before this revision
and will refuse to load -- retrain with `--train 1` instead.

## Metrics: reproducing our results

Every sweep, trace, and plot behind the numbers in this README and in the report lives under
`metrics/`, organized first by agent, then by task:

```
metrics/
├── linear_agent/
│   ├── plot_metrics.py        # generic per-run training-log plotting utility
│   ├── task1/                 # coin-heaven hyperparameter sweep (alpha x gamma grid)
│   ├── task2/                 # loot-crate reward-magnitude sweeps and self-kill investigation
│   ├── task3/                 # peaceful_agent / coin_collector_agent 5-seed sweeps + trace_*.py death/stalemate scripts
│   └── task4/                 # rule_based_agent 5-seed sweep + trace_*.py death/escape scripts
└── tree_agent/
    ├── refit_interval_sweep.py  # re-sweep of REFIT_INTERVAL against Task 2's own workload (negative result)
    ├── task1/                   # coin-heaven hyperparameter sweep (REFIT_INTERVAL x N_ESTIMATORS grid)
    ├── task2/                   # EXTRA_CRATE_BONUS ablation figure
    ├── task3/                   # OPPONENT_POTENTIAL_SCALE regression-fix figure + training learning curve
    └── task4/                   # adjacent-opponent veto fix figure + 4-player FFA comparison
```

Each `plot_*.py` script writes its rendered figures into the matching `results/<agent>/taskN/`
folder (mirroring the `metrics/` layout 1:1):

```
results/
├── linear_agent/
│   ├── task1/   # figures referenced in report1_body.typ
│   ├── task2/   # figures referenced in report2_body.typ
│   ├── task3/   # figures referenced in report3_body.typ
│   └── task4/   # figures referenced in report4_body.typ
└── tree_agent/
    ├── task1/   # figures referenced in report_tree_task1_body.typ
    ├── task2/   # figures referenced in report_tree_task2_body.typ
    ├── task3/   # figures, per-opponent eval JSONs, and shipped models referenced in report_tree_task3_body.typ
    ├── task4/   # figures, eval JSONs, and the shipped model referenced in report_tree_task4_body.typ
    └── refit_interval_sweep/  # per-config training JSONs from the REFIT_INTERVAL re-sweep
```

`results/` itself is gitignored (regenerable, grows large), but the `results/<agent>/taskN/`
figure PNGs specifically are force-added and tracked, since they're the actual images the report
embeds. Every other script writes its own `--save-stats` JSON / CSV / raw training-log output
elsewhere under `results/` (untracked, regenerable by rerunning), and all scripts are self-locating
(resolving the repository root from their own file path, not the working directory you invoke them
from) so they can be run directly from a fresh clone.

### linear_agent

**Task 1** -- hyperparameter sweep behind the Task 1 heatmaps/learning curves:
```bash
uv run metrics/linear_agent/task1/sweep.py        # ~2000 rounds x 16 (alpha, gamma) configurations
uv run metrics/linear_agent/task1/plot_sweep.py   # renders the figures referenced in report1_body.typ
```

**Task 2** -- reward-magnitude sweeps and the self-kill investigation:
```bash
uv run metrics/linear_agent/task2/reward_sweep.py        # first reward-magnitude sweep
uv run metrics/linear_agent/task2/reward_sweep2.py       # follow-up sweep (resumable via reward_sweep2_resume.py)
uv run metrics/linear_agent/task2/killed_self_sweep.py   # self-kill-rate-focused sweep
uv run metrics/linear_agent/task2/plot_task2.py          # renders the figures referenced in report2_body.typ
uv run metrics/linear_agent/task2/plot_report_extras.py  # supplementary figures
```

**Task 3** -- the 5-seed validation sweeps and the `trace_*.py` death/stalemate scripts that
diagnosed each fix attempt (each one drives the environment directly and prints per-step
Q-values, rather than going through `main.py`, for fine-grained debugging):
```bash
bash metrics/linear_agent/task3/seed_sweep.sh                 # seeds 101-105, vs peaceful_agent
bash metrics/linear_agent/task3/seed_sweep_synth_escape.sh     # seeds 301-305, validates the synthetic escape-update fix
bash metrics/linear_agent/task3/seed_sweep_collector.sh        # seeds 201-205, vs coin_collector_agent
bash metrics/linear_agent/task3/seed_sweep_collector_final.sh  # seeds 401-405, final validated coin_collector_agent config
uv run metrics/linear_agent/task3/trace_stalemates.py          # confirms the ~32% coin_collector_agent stalemate rate
uv run metrics/linear_agent/task3/trace_deaths.py              # general death-cause trace
uv run metrics/linear_agent/task3/trace_deaths_phase1.py       # phase-1 (peaceful_agent) death trace
uv run metrics/linear_agent/task3/trace_deaths_collector_v2.py # coin_collector_agent death trace
uv run metrics/linear_agent/task3/plot_task3.py                 # renders the figures referenced in report3_body.typ
```

**Task 4** -- the `rule_based_agent` validation sweep and the `trace_*.py` scripts that diagnosed
the mutual-retaliation self-kill mechanism and (later) the cleared-board stall:
```bash
bash metrics/linear_agent/task4/seed_sweep_rulebased.sh       # seeds 501-505, vs rule_based_agent
uv run metrics/linear_agent/task4/trace_deaths_rulebased.py   # diagnoses the mutual-bombing self-kill mechanism
uv run metrics/linear_agent/task4/trace_adjacent_moment.py    # inspects Q-values at the moment of bombing an adjacent opponent
uv run metrics/linear_agent/task4/trace_cleared_board.py      # inspects Q-values once the board is fully cleared
uv run metrics/linear_agent/task4/plot_task4.py                # renders the figures referenced in report4_body.typ
```

Each seed sweep takes roughly 25-35 minutes per seed (5000 training rounds + a 200-round eval)
on the reference hardware described in `final_project.pdf`; the full Task 3 + Task 4 sweep suite
is several hours end-to-end. The death/stalemate traces are much faster (seconds to low minutes
each), since they stop as soon as they've collected enough example rounds rather than running a
fixed round count.

`plot_task3.py` / `plot_task4.py` read exclusively from already-archived `results/archive_*`
eval and train JSON files (produced by the sweep scripts above) -- they don't retrain anything
themselves, and a missing archive is skipped with a warning rather than raising. The one
exception is each script's `weight_evolution.png` figure, which needs a per-step
`training_log.csv` (the sweep scripts above don't keep this file -- see the training-speed note
under linear_agent's Task 2 for why). To regenerate it:

```bash
# Task 3 (coin_collector_agent, seed 401):
AGENT_SEED=401 uv run main.py play --agents linear_agent coin_collector_agent \
    --train 1 --scenario classic --no-gui --n-rounds 5000
mkdir -p results/archive_task3_collector_weightevo
cp agent_code/linear_agent/metrics/training_log.csv results/archive_task3_collector_weightevo/training_log.csv

# Task 4 (rule_based_agent, seed 505):
AGENT_SEED=505 uv run main.py play --agents linear_agent rule_based_agent \
    --train 1 --scenario classic --no-gui --n-rounds 5000
mkdir -p results/archive_task4_rulebased_weightevo
cp agent_code/linear_agent/metrics/training_log.csv results/archive_task4_rulebased_weightevo/training_log.csv
```

**Important:** `--train 1` overwrites `agent_code/linear_agent/linear-model.pt` with whatever
that run produces. Immediately after copying out `training_log.csv`, restore the shipped model
before doing anything else, e.g. from the seed sweep archive:
```bash
cp results/archive_task4_rulebased_seed_sweep/seed_505/model.pt agent_code/linear_agent/linear-model.pt
```
Also note `AGENT_SEED` only seeds the agent's own exploration -- it does not reproduce the
original run's exact crate/coin layout (that's controlled by a separate, unused `--seed` flag),
so a regenerated `training_log.csv` will be a qualitatively similar but not byte-identical
training run to the one the shipped model/archived eval numbers came from.

### tree_agent

**Task 1** -- hyperparameter sweep behind the Task 1 heatmaps/learning curves:
```bash
uv run metrics/tree_agent/task1/sweep.py        # ~2000 rounds x 16 (REFIT_INTERVAL, N_ESTIMATORS) configurations
uv run metrics/tree_agent/task1/plot_sweep.py   # renders the figures referenced in report_tree_task1_body.typ
```

**Task 2** -- the per-step budget trace behind the routing revision (runs N pure-greedy rounds
of the installed model in-process and reports where the 400 steps go: crate walking, coin
chasing, escaping, waiting, bombing; steps per bomb cycle; full clears; stuck rounds), the
agent's unit tests, and the `EXTRA_CRATE_BONUS` ablation figure (reads from the already-saved
`results/tree_agent/task2/task2_final_eval_*.json` evaluation files; does not retrain anything
itself):
```bash
uv run python metrics/tree_agent/task2/trace_step_budget.py --rounds 30   # step budget of the installed tree-model.pt on loot-crate
uv run python -m unittest agent_code.tree_agent.tests.test_targeting -v  # target selection, features 11-15, move_bonus
uv run metrics/tree_agent/task2/plot_task2.py   # renders config_comparison_bar.png, referenced in report_tree_task2_body.typ
```

**Task 3** -- the `OPPONENT_POTENTIAL_SCALE` regression-fix figure and the `coin_collector_agent`
training learning curve (reads from the already-saved `results/tree_agent/task3/task3_peaceful_eval*.json`
evaluation files and the archived `training_log_collector.csv`; does not retrain anything itself).
The per-opponent `--save-stats` evaluation JSONs and shipped models referenced in
`report_tree_task3_body.typ` are archived directly under `results/tree_agent/task3/`:
```bash
uv run metrics/tree_agent/task3/plot_task3.py   # renders opponent_potential_scale_fix.png and learning_curve.png, referenced in report_tree_task3_body.typ
```

**Task 4** -- the adjacent-opponent veto fix comparison and the 4-player free-for-all comparison
(reads from the already-saved `results/tree_agent/task4/*.json` evaluation files; does not retrain
anything itself):
```bash
uv run metrics/tree_agent/task4/plot_task4.py   # renders adjacent_opponent_veto_fix.png, ffa_comparison.png, and regression_check.png, referenced in report_tree_task4_body.typ
```

**The `REFIT_INTERVAL` re-sweep** -- re-tests the Task 1 hyperparameter sweep's `REFIT_INTERVAL`
choice against Task 2's own (much longer) round-length distribution. A negative result: 10 looked
like a clear win in training-time metrics but caused a catastrophic pure-greedy policy collapse a
full retrain + evaluation exposed -- reverted, 25 confirmed correct. See `train.py`'s
`REFIT_INTERVAL` comment for the full account.
```bash
uv run metrics/tree_agent/refit_interval_sweep.py   # ~2000 rounds x 4 REFIT_INTERVAL configurations on loot-crate
```

**Important:** as with `linear_agent`, `--train 1` overwrites `agent_code/tree_agent/tree-model.pt`
with whatever that run produces -- back up the shipped model first if you need to restore it
afterward. The shipped `tree-model.pt` is the Task 2 routing model; the per-opponent models
archived under `results/tree_agent/task3/` and `task4/` predate the routing revision (11
features) and cannot be loaded by the current code -- retrain before using them.
