"""
Which features does the shipped Q-function actually use?

Two measures, because they disagree in a way that matters. Impurity importance
is what scikit-learn reports by default, and it is biased toward features with
many distinct split points -- exactly the situation here, where feature 11 is
near-continuous and the rest are binary or small integers (Breiman 2001;
Strobl et al. 2007). Permutation importance on held-out data does not share that
bias, so it is the one to trust for "does the model use this feature".

Reads the archived replay buffer; trains nothing and writes nothing to
agent_code/.

Usage (from the repository root):
    python metrics/tree_agent/feature_importance.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.inspection import permutation_importance

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

MODEL = REPO_ROOT / "agent_code" / "tree_agent" / "tree-model.pt"
REPLAY = REPO_ROOT / "results" / "tree_agent" / "task4" / "replay_shipped_model_250rounds.pkl"
OUT = REPO_ROOT / "results" / "tree_agent" / "analysis"

NAMES = [
    "moves_to_coin", "valid", "is_wait", "is_bomb", "into_danger",
    "escape_if_bomb", "crates_if_bomb", "escape_move", "moves_to_crate",
    "moves_to_opponent", "hits_opponent", "target_dist", "target_delta",
    "yield_here", "cooldown", "target_is_coin",
]


def main(n_sample=20000, n_repeats=5, seed=0):
    from agent_code.tree_agent.features import GAMMA, ACTIONS, N_FEATURES

    model = pickle.load(open(MODEL, "rb"))
    replay = pickle.load(open(REPLAY, "rb"))
    print(f"model: {type(model).__name__}, {len(model.estimators_)} trees, "
          f"{model.n_features_in_} features")
    print(f"replay: {len(replay)} transitions")

    phis = np.stack([t[0] for t in replay])
    rewards = np.array([t[1] for t in replay])
    next_phis = np.stack([t[2] for t in replay])
    done = np.array([t[3] for t in replay])

    # the regression target the model was actually fitted on
    flat = next_phis.reshape(-1, N_FEATURES)
    next_q = model.predict(flat).reshape(len(replay), len(ACTIONS))
    max_next = next_q.max(axis=1)
    max_next[done] = 0.0
    targets = rewards + GAMMA * max_next

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(phis), size=min(n_sample, len(phis)), replace=False)
    X, y = phis[idx], targets[idx]

    print(f"permutation importance on {len(X)} held-out-style rows, "
          f"{n_repeats} repeats ...", flush=True)
    perm = permutation_importance(model, X, y, n_repeats=n_repeats,
                                  random_state=seed, scoring="neg_root_mean_squared_error")

    imp = model.feature_importances_
    rows = []
    for i in range(N_FEATURES):
        rows.append({
            "index": i, "name": NAMES[i],
            "impurity": float(imp[i]),
            "permutation_mean": float(perm.importances_mean[i]),
            "permutation_std": float(perm.importances_std[i]),
            "n_distinct_values": int(len(np.unique(phis[:, i]))),
        })
    rows.sort(key=lambda r: -r["permutation_mean"])

    print(f"\n{'#':>3} {'feature':<20}{'impurity':>10}{'permutation ΔRMSE':>21}{'distinct':>10}")
    for r in rows:
        print(f"{r['index']:>3} {r['name']:<20}{r['impurity']:>10.4f}"
              f"{r['permutation_mean']:>14.4f} ±{r['permutation_std']:<5.4f}{r['n_distinct_values']:>10}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "feature_importance.json").write_text(json.dumps({
        "model": str(MODEL.relative_to(REPO_ROOT)),
        "replay": str(REPLAY.relative_to(REPO_ROOT)),
        "n_transitions": len(replay),
        "n_sampled": len(X),
        "n_repeats": n_repeats,
        "scoring": "neg_root_mean_squared_error",
        "features": rows,
    }, indent=2))
    print(f"\n-> {OUT / 'feature_importance.json'}")


if __name__ == "__main__":
    main()
