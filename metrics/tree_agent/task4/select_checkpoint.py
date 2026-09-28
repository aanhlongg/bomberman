import argparse
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 9, "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
})


def evaluate(agent_name, model_path, opponents, rounds, seed, out_json):
    """One `main.py play` evaluation of `model_path` as agent `agent_name`; returns the stats dict."""
    if not Path(out_json).exists():
        env = dict(os.environ, TREE_AGENT_MODEL_PATH=str(Path(model_path).resolve()))
        cmd = ["uv", "run", "main.py", "play", "--agents", agent_name, *opponents, "--scenario", "classic",
               "--no-gui", "--n-rounds", str(rounds), "--seed", str(seed), "--save-stats", str(out_json)]
        subprocess.run(cmd, env=env, check=True, capture_output=True)
    d = json.load(open(out_json))
    me = d["by_agent"][agent_name]; n = me["rounds"]
    others = [a for k, a in d["by_agent"].items() if k != agent_name]
    return {
        "rounds": n, "score": me.get("score", 0), "opp_score": sum(a.get("score", 0) for a in others) / max(1, len(others)),
        "kills": me.get("kills", 0), "suicides": me.get("suicides", 0), "coins": me.get("coins", 0),
        "margin_per_round": (me.get("score", 0) - sum(a.get("score", 0) for a in others) / max(1, len(others))) / n,
        "suicide_rate": me.get("suicides", 0) / n, "kill_rate": me.get("kills", 0) / n,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help='"label=agent_dir" (repeatable); agent_dir must be under agent_code/')
    ap.add_argument("--opponent", default="rule_based_agent")
    ap.add_argument("--rounds1", type=int, default=50)
    ap.add_argument("--rounds2", type=int, default=200)
    ap.add_argument("--seed1", type=int, default=1)
    ap.add_argument("--seed2", type=int, default=2)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--max-suicide-rate", type=float, default=0.12)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--workdir", default="results/tree_agent/task4/checkpoint_evals")
    ap.add_argument("--out", default="results/tree_agent/task4/greedy_vs_training_round.png")
    a = ap.parse_args()
    work = Path(a.workdir); work.mkdir(parents=True, exist_ok=True)

    jobs = []
    for spec in a.run:
        label, path = spec.split("=", 1)
        agent_name = Path(path).name
        for ckpt in sorted(Path(path, "checkpoints").glob("model_round_*.pt"), key=lambda p: int(re.findall(r"\d+", p.name)[0])):
            rnd = int(re.findall(r"\d+", ckpt.name)[0])
            jobs.append((label, agent_name, rnd, ckpt))
    print(f"stage 1: {len(jobs)} checkpoints x {a.rounds1} rounds vs {a.opponent} (seed {a.seed1}), {a.jobs} in parallel")
    stage1 = {}
    with ThreadPoolExecutor(a.jobs) as pool:
        futs = {pool.submit(evaluate, agent_name, ckpt, [a.opponent], a.rounds1, a.seed1,
                            work / f"s1_{agent_name}_r{rnd}.json"): (label, rnd) for label, agent_name, rnd, ckpt in jobs}
        for fut, key in futs.items():
            stage1[key] = fut.result()
    for (label, rnd), r in sorted(stage1.items()):
        print(f"  {label:8} round {rnd:6d}: margin {r['margin_per_round']:+6.2f}/rd  score {r['score']:5d} vs {r['opp_score']:6.0f}  kills {r['kill_rate']:5.1%}  suicides {r['suicide_rate']:5.1%}")

    eligible = [(k, r) for k, r in stage1.items() if r["suicide_rate"] <= a.max_suicide_rate]
    ranked = sorted(eligible, key=lambda kr: kr[1]["margin_per_round"], reverse=True)[: a.top]
    print(f"\nstage 2: top {len(ranked)} (suicide rate <= {a.max_suicide_rate:.0%}) x {a.rounds2} fresh rounds (seed {a.seed2}) + FFA")

    stage2 = {}
    lookup = {(label, rnd): (agent_name, ckpt) for label, agent_name, rnd, ckpt in jobs}
    with ThreadPoolExecutor(a.jobs) as pool:
        futs = {}
        for key, _ in ranked:
            agent_name, ckpt = lookup[key]
            futs[pool.submit(evaluate, agent_name, ckpt, [a.opponent], a.rounds2, a.seed2, work / f"s2_{agent_name}_r{key[1]}.json")] = (key, "1v1")
            if a.opponent == "rule_based_agent":
                futs[pool.submit(evaluate, agent_name, ckpt, [a.opponent] * 3, a.rounds2, a.seed2, work / f"s2ffa_{agent_name}_r{key[1]}.json")] = (key, "ffa")
        for fut, (key, kind) in futs.items():
            stage2.setdefault(key, {})[kind] = fut.result()
    for key, res in sorted(stage2.items(), key=lambda kv: kv[1]["1v1"]["margin_per_round"], reverse=True):
        r = res["1v1"]; f = res.get("ffa")
        print(f"  {key[0]:8} round {key[1]:6d}: 1v1 margin {r['margin_per_round']:+6.2f}/rd  score {r['score']:5d} vs {r['opp_score']:6.0f}  kills {r['kill_rate']:5.1%}  suicides {r['suicide_rate']:5.1%}"
              + (f"   | FFA score {f['score']:5d} (avg opp {f['opp_score']:5.0f}) kills {f['kill_rate']:5.1%} suicides {f['suicide_rate']:5.1%}" if f else ""))

    json.dump({"stage1": {f"{k[0]}@{k[1]}": v for k, v in stage1.items()}, "stage2": {f"{k[0]}@{k[1]}": v for k, v in stage2.items()}},
              open(work / "summary.json", "w"), indent=2)

    labels = sorted({k[0] for k in stage1})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 6.4), sharex=True)
    for label, color in zip(labels, [BLUE, ORANGE, AQUA, YELLOW]):
        pts = sorted((k[1], r) for k, r in stage1.items() if k[0] == label)
        xs = [p[0] for p in pts]
        ax1.plot(xs, [p[1]["margin_per_round"] for p in pts], color=color, linewidth=1.5, marker="o", markersize=3.5, label=f"{label}, stage 1 ({a.rounds1} rounds each)")
        ax2.plot(xs, [p[1]["suicide_rate"] for p in pts], color=color, linewidth=1.5, marker="o", markersize=3.5, label=label)
        s2 = [(k[1], v["1v1"]) for k, v in stage2.items() if k[0] == label]
        if s2:
            ax1.scatter([p[0] for p in s2], [p[1]["margin_per_round"] for p in s2], color=color, edgecolor=INK, s=60, zorder=5,
                        label=f"{label}, stage 2 ({a.rounds2} fresh rounds)")
            ax2.scatter([p[0] for p in s2], [p[1]["suicide_rate"] for p in s2], color=color, edgecolor=INK, s=60, zorder=5)
    ax1.axhline(0, color=MUTED, linewidth=0.8)
    ax1.set_ylabel(f"score margin per round\n(ours - {a.opponent})")
    ax1.set_title(f"Pure-greedy quality of every checkpoint vs {a.opponent}", loc="left")
    ax1.legend(frameon=False, fontsize=8)
    ax2.axhline(a.max_suicide_rate, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    ax2.text(ax2.get_xlim()[0], a.max_suicide_rate, f" stage-2 cap {a.max_suicide_rate:.0%}", va="bottom", color=INK_2, fontsize=8.5)
    ax2.set_ylabel("own-bomb suicide rate")
    ax2.set_xlabel("training round of the checkpoint")
    for ax in (ax1, ax2):
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print("\nwrote", a.out, "and", work / "summary.json")


if __name__ == "__main__":
    main()
