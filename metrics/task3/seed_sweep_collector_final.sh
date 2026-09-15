#!/bin/bash
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SEEDS="401 402 403 404 405"

for seed in $SEEDS; do
    echo "=== SEED $seed: training (5000 rounds) ==="
    AGENT_SEED=$seed uv run main.py play --agents linear_agent coin_collector_agent \
        --train 1 --scenario classic --no-gui --n-rounds 5000 \
        --save-stats "results/task3_collector_finalseed_${seed}_train.json"

    cp agent_code/linear_agent/linear-model.pt "results/task3_collector_finalseed_${seed}_model.pt"
    cp agent_code/linear_agent/metrics/training_log.csv "results/task3_collector_finalseed_${seed}_training_log.csv"

    echo "=== SEED $seed: eval (200 rounds, pure greedy) ==="
    AGENT_SEED=$seed uv run main.py play --agents linear_agent coin_collector_agent \
        --scenario classic --no-gui --n-rounds 200 \
        --save-stats "results/task3_collector_finalseed_${seed}_eval.json"

    echo "=== SEED $seed: done ==="
done

echo "ALL SEEDS DONE"
