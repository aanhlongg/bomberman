<div align="center">
    <h1> Reinforcement Learning Agents in Bomberman </h1>
    <h3></h3>
</div>

## Linear agent: Task 1 (coin-heaven)

train the linear agent:
```bash
uv run main.py play --agents linear_agent --train 1 --scenario coin-heaven --no-gui --n-rounds 1000
```

evaluate the linear agent:
```bash
uv run main.py play --agents linear_agent --scenario coin-heaven # single run 
uv run main.py play --agents linear_agent --scenario coin-heaven --no-gui --n-rounds 200 # multiple runs
```

## Linear agent: Task 2 (loot-crate)

train the linear agent:
```bash
# training will average the weight vector starting at round 1000
uv run main.py play --agents linear_agent --train 1 --scenario loot-crate --no-gui --n-rounds 3000
```

evaluate the linear agent:
```bash
uv run main.py play --agents linear_agent --scenario loot-crate  # single run 
uv run main.py play --agents linear_agent --scenario loot-crate --no-gui --n-rounds 200 # multiple runs
```
