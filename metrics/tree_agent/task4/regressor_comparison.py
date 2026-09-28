import argparse
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from agent_code.tree_agent import train as T
from agent_code.tree_agent.features import ACTIONS, GAMMA, N_FEATURES

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
})

MONOTONE = np.zeros(N_FEATURES, dtype=int)
MONOTONE[4] = -1
MONOTONE[5] = +1
MONOTONE[6] = +1
MONOTONE[11] = -1
MONOTONE[13] = +1


def candidates(seed):
    return {
        "ExtraTrees (shipped)": lambda: ExtraTreesRegressor(
            n_estimators=T.N_ESTIMATORS, min_samples_leaf=T.MIN_SAMPLES_LEAF, random_state=seed),
        "HGB conservative": lambda: HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0, random_state=seed),
        "HGB sharp": lambda: HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20, random_state=seed),
        "HGB conservative + monotone": lambda: HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0,
            monotonic_cst=MONOTONE, random_state=seed),
    }


def bellman_targets(model, rewards, next_phis, done):
    if model is None:
        return rewards.copy()
    q = model.predict(next_phis.reshape(-1, N_FEATURES)).reshape(len(rewards), len(ACTIONS))
    mx = q.max(axis=1)
    mx[done] = 0.0
    return rewards + GAMMA * mx


def action_gaps(model, next_phis):
    q = model.predict(next_phis.reshape(-1, N_FEATURES)).reshape(len(next_phis), len(ACTIONS))
    top2 = np.sort(q, axis=1)[:, -2:]
    return q.argmax(axis=1), top2[:, 1] - top2[:, 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--model", default="agent_code/tree_agent/tree-model.pt")
    ap.add_argument("--sweeps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/tree_agent/task4/regressor_comparison.png")
    a = ap.parse_args()

    replay = pickle.load(open(a.replay, "rb"))
    shipped = pickle.load(open(a.model, "rb"))
    phis = np.stack([t[0] for t in replay]); rewards = np.array([t[1] for t in replay], dtype=float)
    next_phis = np.stack([t[2] for t in replay]); done = np.array([t[3] for t in replay], dtype=bool)
    n = len(replay)
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(n); cut = int(0.8 * n); tr, te = idx[:cut], idx[cut:]
    targets = bellman_targets(shipped, rewards, next_phis, done)
    live = te[~done[te]]
    print(f"buffer: {n:,} transitions ({done.sum()} terminal), {len(tr):,} train / {len(te):,} held out; "
          f"targets mean {targets.mean():.2f} sd {targets.std():.2f}\n")

    ship_argmax, ship_gap = action_gaps(shipped, next_phis[live])
    results = {}
    print(f"{'regressor':30} {'RMSE':>7} {'argmax agree':>13} {'median gap':>11} {'gap<0.5':>8} {'fit s':>7} {'predict ms':>11}")
    for name, make in candidates(a.seed).items():
        m = make()
        t0 = time.perf_counter(); m.fit(phis[tr], targets[tr]); fit_s = time.perf_counter() - t0
        rmse = float(np.sqrt(np.mean((m.predict(phis[te]) - targets[te]) ** 2)))
        am, gap = action_gaps(m, next_phis[live])
        agree = float(np.mean(am == ship_argmax))
        six = next_phis[live[0]]
        t0 = time.perf_counter()
        for _ in range(200):
            m.predict(six)
        pred_ms = (time.perf_counter() - t0) / 200 * 1000
        results[name] = dict(rmse=rmse, agree=agree, gap=gap, fit_s=fit_s, pred_ms=pred_ms)
        print(f"{name:30} {rmse:7.3f} {agree:13.1%} {np.median(gap):11.2f} {np.mean(gap < 0.5):8.1%} {fit_s:7.1f} {pred_ms:11.2f}")
    print(f"{'(shipped model itself)':30} {'':>7} {'':>13} {np.median(ship_gap):11.2f} {np.mean(ship_gap < 0.5):8.1%}")

    print(f"\nQ-inflation over {a.sweeps} FQI sweeps (mean / max predicted Q on the buffer):")
    inflation = {}
    for name, make in candidates(a.seed).items():
        model = None; means, maxes = [], []
        for _ in range(a.sweeps):
            tg = bellman_targets(model, rewards, next_phis, done)
            model = make(); model.fit(phis, tg)
            q = model.predict(phis); means.append(q.mean()); maxes.append(q.max())
        inflation[name] = (means, maxes)
        print(f"  {name:30} mean: {' '.join(f'{v:6.1f}' for v in means)}")
        print(f"  {'':30} max : {' '.join(f'{v:6.1f}' for v in maxes)}")
    print(f"  reference: mean reward {rewards.mean():.2f} -> r/(1-gamma) = {rewards.mean()/(1-GAMMA):.1f}")

    names = list(results); colors = [MUTED, BLUE, ORANGE, AQUA]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    ax = axes[0, 0]
    ax.bar(range(len(names)), [results[k]["rmse"] for k in names], color=colors, width=0.6, edgecolor=SURFACE, linewidth=2)
    for i, k in enumerate(names):
        ax.text(i, results[k]["rmse"], f"{results[k]['rmse']:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(names))); ax.set_xticklabels([k.replace(" + ", "\n+ ").replace(" (", "\n(") for k in names], fontsize=8)
    ax.set_ylabel("held-out RMSE on Bellman targets"); ax.set_title("Fit quality (lower is better)", loc="left"); ax.grid(axis="x", visible=False)
    ax = axes[0, 1]
    ax.bar(range(len(names)), [results[k]["agree"] for k in names], color=colors, width=0.6, edgecolor=SURFACE, linewidth=2)
    for i, k in enumerate(names):
        ax.text(i, results[k]["agree"], f"{results[k]['agree']:.0%}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(names))); ax.set_xticklabels([k.replace(" + ", "\n+ ").replace(" (", "\n(") for k in names], fontsize=8)
    ax.set_ylim(0, 1.08); ax.set_ylabel("share of next states with the same greedy action"); ax.set_title("Argmax agreement with the shipped model", loc="left"); ax.grid(axis="x", visible=False)
    ax = axes[1, 0]
    bins = np.linspace(0, 8, 41)
    for k, c in zip(names, colors):
        ax.hist(np.clip(results[k]["gap"], 0, 8), bins=bins, histtype="step", linewidth=1.6, color=c, label=k, density=True)
    ax.set_xlabel("action gap Q(best) - Q(second) on held-out next states (clipped at 8)"); ax.set_ylabel("density")
    ax.set_title("How decisively actions are separated", loc="left"); ax.legend(frameon=False); ax.grid(axis="x", visible=False)
    ax = axes[1, 1]
    for k, c in zip(names, colors):
        means, maxes = inflation[k]
        ax.plot(range(1, a.sweeps + 1), means, color=c, linewidth=1.8, label=f"{k} (mean)")
        ax.plot(range(1, a.sweeps + 1), maxes, color=c, linewidth=1.0, linestyle=(0, (3, 2)))
    ax.set_xlabel("FQI sweep on the fixed buffer (solid: mean Q, dashed: max Q)"); ax.set_ylabel("predicted Q")
    ax.set_title("Q-inflation: does repeated bootstrapping saturate?", loc="left"); ax.legend(frameon=False, fontsize=8); ax.grid(axis="x", visible=False)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
