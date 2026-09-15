<div align="center">
    <h1> Reinforcement Learning Agents in Bomberman </h1>
    <h3></h3>
</div>

## Linear agent: Task 1 (collecting coins)

train the linear agent:
```bash
uv run main.py play --agents linear_agent --train 1 --scenario coin-heaven --no-gui --n-rounds 1000
```

evaluate the linear agent:
```bash
uv run main.py play --agents linear_agent --scenario coin-heaven # single run 
uv run main.py play --agents linear_agent --scenario coin-heaven --no-gui --n-rounds 200 # multiple runs
```

## Linear agent: Task 2 (clearing crates)

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

## Linear agent: Task 3 (hunting opponents)

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
— see Task 4 below; it also handles Task 3's opponents well as an emergent side effect, see
`report3_body.typ` / `report4_body.typ`), averaged over 200 evaluation rounds:

| Opponent | Score (ours / theirs) | Our kills | Our suicides |
|---|---|---|---|
| `peaceful_agent` | 2550 / 4 | 154 (77%) | 5 (2.5%) |
| `coin_collector_agent` | 1299 / 654 | 29 (14.5%) | 31 (15.5%) |

`coin_collector_agent`'s stalemate rate (rounds that end with the board fully cleared but neither
agent finishing the other off) is a known, documented limitation — see the "Open Questions"
section of `report3_body.typ` for the full investigation and every fix attempted.

## Linear agent: Task 4 (holding your own against rule_based_agent)

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
score 1220-1280 vs. 712-794 across all 5 seeds), not just this single run — see
`report4_body.typ` for the full sweep table and every fix attempted (including several that
were tried and reverted after regressing other opponents).

## Metrics: reproducing our results

Every sweep, trace, and plot behind the numbers in this README and in the report lives under
`metrics/`, organized first by agent, then by task (`tree_agent` will get its own sibling
folder here as its tasks are completed):

```
metrics/
└── linear_agent/
    ├── plot_metrics.py        # generic per-run training-log plotting utility
    ├── task1/                 # coin-heaven hyperparameter sweep (alpha x gamma grid)
    ├── task2/                 # loot-crate reward-magnitude sweeps and self-kill investigation
    ├── task3/                 # peaceful_agent / coin_collector_agent 5-seed sweeps + death/stalemate traces
    └── task4/                 # rule_based_agent 5-seed sweep + death/escape traces
```

All scripts are self-locating (they resolve the repository root from their own file path, not
from the working directory you invoke them from), so they can be run directly from a fresh
clone. Each script writes its own `--save-stats` JSON / CSV output into `results/` (created if
missing; gitignored, since it's regenerable and grows large) rather than overwriting anything
tracked in git.

**Task 1** — hyperparameter sweep behind the Task 1 heatmaps/learning curves:
```bash
uv run metrics/linear_agent/task1/sweep.py        # ~2000 rounds x 16 (alpha, gamma) configurations
uv run metrics/linear_agent/task1/plot_sweep.py   # renders the figures referenced in report1_body.typ
```

**Task 2** — reward-magnitude sweeps and the self-kill investigation:
```bash
uv run metrics/linear_agent/task2/reward_sweep.py        # first reward-magnitude sweep
uv run metrics/linear_agent/task2/reward_sweep2.py       # follow-up sweep (resumable via reward_sweep2_resume.py)
uv run metrics/linear_agent/task2/killed_self_sweep.py   # self-kill-rate-focused sweep
uv run metrics/linear_agent/task2/plot_task2.py          # renders the figures referenced in report2_body.typ
uv run metrics/linear_agent/task2/plot_report_extras.py  # supplementary figures
```

**Task 3** — the 5-seed validation sweeps and the death/stalemate traces that diagnosed each
fix attempt (each trace script drives the environment directly and prints per-step Q-values,
rather than going through `main.py`, for fine-grained debugging):
```bash
bash metrics/linear_agent/task3/seed_sweep.sh                 # seeds 101-105, vs peaceful_agent
bash metrics/linear_agent/task3/seed_sweep_synth_escape.sh     # seeds 301-305, validates the synthetic escape-update fix
bash metrics/linear_agent/task3/seed_sweep_collector.sh        # seeds 201-205, vs coin_collector_agent
bash metrics/linear_agent/task3/seed_sweep_collector_final.sh  # seeds 401-405, final validated coin_collector_agent config
uv run metrics/linear_agent/task3/trace_stalemates.py          # confirms the ~32% coin_collector_agent stalemate rate
uv run metrics/linear_agent/task3/trace_deaths.py              # general death-cause trace
uv run metrics/linear_agent/task3/trace_deaths_phase1.py       # phase-1 (peaceful_agent) death trace
uv run metrics/linear_agent/task3/trace_deaths_collector_v2.py # coin_collector_agent death trace
```

**Task 4** — the `rule_based_agent` validation sweep and the traces that diagnosed the
mutual-retaliation self-kill mechanism and (later) the cleared-board stall:
```bash
bash metrics/linear_agent/task4/seed_sweep_rulebased.sh       # seeds 501-505, vs rule_based_agent
uv run metrics/linear_agent/task4/trace_deaths_rulebased.py   # diagnoses the mutual-bombing self-kill mechanism
uv run metrics/linear_agent/task4/trace_adjacent_moment.py    # inspects Q-values at the moment of bombing an adjacent opponent
uv run metrics/linear_agent/task4/trace_cleared_board.py      # inspects Q-values once the board is fully cleared
```

Each seed sweep takes roughly 25-35 minutes per seed (5000 training rounds + a 200-round eval)
on the reference hardware described in `final_project.pdf`; the full Task 3 + Task 4 sweep suite
is several hours end-to-end. The death/stalemate traces are much faster (seconds to low minutes
each), since they stop as soon as they've collected enough example rounds rather than running a
fixed round count.
