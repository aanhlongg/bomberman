# Commands:
```bash
# train the initial coin-collector-agent in coin-heaven
uv run main.py play --no-gui --agents agent_pgk --train 1 --scenario coin-heaven --n-rounds 400
```
```bash
# train, evaluate the initial coin-collector-agent in coin heaven and display the evaluation plot
uv run evaluate.py --agent agent_pgk --n-rounds 400 --scenario coin-heaven --show
```
```bash
# test the initial coin-collector-agent in coin heaven 
uv run main.py play --my-agent agent_pgk --scenario coin-heaven
```