# Commands:
```bash
# The initial coin-collector-agent "agent_pgk" got trained in coin-heaven using the following command 
uv run main.py play --no-gui --agents agent_pgk rule_based_agent --train 1 --scenario coin-heaven --n-rounds 1000
```
```bash
# This command runs a specified number of simulations (here 300) of test-rounds and plots them
uv run evaluate.py --agent agent_pgk --eval --n-rounds 300 --scenario coin-heaven --show
```
```bash
# test the initial coin-collector-agent in coin heaven 
uv run main.py play --agent agent_pgk --scenario coin-heaven
```
