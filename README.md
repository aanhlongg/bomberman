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
