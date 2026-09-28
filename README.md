<div align="center">
    <h1> Reinforcement Learning Agents in Bomberman </h1>
</div>

### Requirements

Requires [uv](https://docs.astral.sh/uv/).

<details>
<summary><h2>Linear agent (<code>linear_agent</code>)</h2></summary>

Produces a single shipped model `agent_code/linear_agent/linear-model.pt` which is used for all four tasks.

### Train

Training starts from zero weights and writes a checkpoint every 100 rounds to
`agent_code/linear_agent/metrics/checkpoints/`.

```bash
# task 1: coin-heaven
uv run main.py play --agents linear_agent --train 1 --scenario coin-heaven --no-gui --n-rounds 1000

# task 2: loot-crate
uv run main.py play --agents linear_agent --train 1 --scenario loot-crate --no-gui --n-rounds 1500

# tasks 3 and 4: the shipped model was trained with this command
uv run main.py play --agents linear_agent peaceful_agent coin_collector_agent --train 1 --scenario classic --no-gui --n-rounds 1500
```

### Choose the checkpoint

Training does not converge, so the model is selected from the checkpoints rather
than taken from the end of the run. The `linear-model.pt` that training itself
writes is a running weight average and should not be used.

```bash
# evaluates every checkpoint and prints the best one with the command to install it
uv run metrics/sweep_checkpoints.py --scenario coin-heaven --jobs 8                                      # task 1
uv run metrics/sweep_checkpoints.py --scenario loot-crate --jobs 8                                       # task 2
uv run metrics/sweep_checkpoints.py --opponents peaceful_agent coin_collector_agent --jobs 8             # tasks 3 and 4

# then install the round it names, e.g.
cp agent_code/linear_agent/metrics/checkpoints/weights_600.pt agent_code/linear_agent/linear-model.pt
```

### Evaluate

Loads `linear-model.pt` and plays greedily, without exploration or training.

```bash
uv run metrics/evaluate_model.py --agent linear_agent --scenario coin-heaven --n-rounds 200               # task 1
uv run metrics/evaluate_model.py --agent linear_agent --scenario loot-crate --n-rounds 200                # task 2
uv run metrics/evaluate_task3.py --opponents peaceful_agent coin_collector_agent --n-rounds 200           # task 3
uv run metrics/evaluate_task3.py --opponents rule_based_agent --n-rounds 200                             # task 4

# watch a single round in the GUI
uv run main.py play --agents linear_agent rule_based_agent --scenario classic
```

</details>

<details>
<summary><h2>Tree agent (<code>tree_agent</code>)</h2></summary>

Produces a single shipped model `agent_code/tree_agent/tree-model.pt` which is used for all four tasks.

### Train

Training starts from scratch and refits the model every 25 rounds, overwriting
`agent_code/tree_agent/tree-model.pt` each time, so back up the shipped model first.
The model from the final refit is the one that ships. A 5000-round run takes about
1.5 hours, and separate runs vary a lot in strength, so the shipped model is the best
of several runs.

```bash
# back up the shipped model
cp agent_code/tree_agent/tree-model.pt agent_code/tree_agent/tree-model.shipped.pt

# all four tasks: the shipped model was trained with this command
uv run main.py play --agents tree_agent rule_based_agent --train 1 --scenario classic --no-gui --n-rounds 5000
```

### Evaluate

Loads `tree-model.pt` and plays greedily, without exploration or training.

```bash
uv run metrics/evaluate_model.py --agent tree_agent --scenario coin-heaven --n-rounds 200                               # task 1
uv run metrics/evaluate_model.py --agent tree_agent --scenario loot-crate --n-rounds 200                                # task 2
uv run metrics/evaluate_task3.py --agent tree_agent --opponents peaceful_agent coin_collector_agent --n-rounds 200      # task 3
uv run metrics/evaluate_task3.py --agent tree_agent --opponents rule_based_agent --n-rounds 200                        # task 4

# watch a single round in the GUI
uv run main.py play --agents tree_agent rule_based_agent --scenario classic
```

</details>
