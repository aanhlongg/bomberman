"""
plot_head_to_head.py -- the figure for section 6.4 (tree_agent vs rcp-linear)

Run from the repo root:

    uv run python metrics/tree_agent/plot_head_to_head.py

Section 6.4 rests on two readings that disagree, and the point of the figure is
that they are conditional rather than contradictory. Both are already on disk;
nothing is replayed here:

  results/tree_agent/campaign/head_to_head/seed10??.json
      -- ten seeded 50-round matches on `classic`, tree_agent + rcp-linear +
         rule_based_agent, same boards for both agents (seeded_campaign.py)
  results/tree_agent/tournament_2026-09-25/tournament_2026-09-25.json
      -- the one recorded four-player tournament, 150 rounds on a coin-rich
         board, recovered from logs/game.log by parse_game_log.py because the
         match was run without --save-stats. n = 1, and the caption says so.

Renders one figure with three panels, each on a single axis:

  1. score per round on `classic`, per agent, mean over the ten matches
  2. score per round in the tournament, per agent, mean over the 150 rounds
  3. own-bomb deaths per round in both settings

Panels 1 and 2 carry different boards and are not comparable in absolute terms,
which is why they are separate axes rather than one grouped chart.

Output: results/tree_agent/analysis/head_to_head.png
Every printed number is checked against section 6.4 in the report.
"""

import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = REPO_ROOT / "results" / "tree_agent" / "campaign" / "head_to_head"
TOURNAMENT = (REPO_ROOT / "results" / "tree_agent" / "tournament_2026-09-25"
              / "tournament_2026-09-25.json")
OUT = REPO_ROOT / "results" / "tree_agent" / "analysis"

TREE, RCP = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b2"

AGENTS = ("tree_agent", "rcp-linear")
COLOR = {"tree_agent": TREE, "rcp-linear": RCP}


def ci95(x):
    """Mean and half-width of a 95 % CI; same helper as analyze_campaign.py."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else float("nan"), 0.0
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def style(ax):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.8)
    ax.grid(True, axis="x", color=MUTED, alpha=0.35, linewidth=0.6)
    ax.set_axisbelow(True)


def load_classic():
    """Per-match score and own-bomb deaths per round, one row per seeded match."""
    out = {a: {"score": [], "suicides": []} for a in AGENTS}
    files = sorted(glob.glob(str(CAMPAIGN / "seed*.json")))
    if not files:
        raise SystemExit(f"no seeded matches under {CAMPAIGN}")
    for path in files:
        by_agent = json.load(open(path))["by_agent"]
        for a in AGENTS:
            rec = by_agent[a]
            n = rec["rounds"]
            out[a]["score"].append(rec["score"] / n)
            out[a]["suicides"].append(rec["suicides"] / n)
    return out, len(files)


def load_tournament():
    """Per-round score and own-bomb deaths, one row per round, per agent."""
    by_agent_round = json.load(open(TOURNAMENT))["by_agent_round"]
    out = {}
    for a in AGENTS:
        rounds = by_agent_round[a]
        out[a] = {
            "score": np.array([r["score"] for r in rounds], dtype=float),
            "suicides": np.array([r["suicides"] for r in rounds], dtype=float),
        }
    return out, len(by_agent_round[AGENTS[0]])


def dot_panel(ax, values, title, subtitle, xlabel):
    """One dot per agent with a 95 % CI whisker; identity carried by color + label."""
    for i, a in enumerate(AGENTS):
        m, e = ci95(values[a])
        y = len(AGENTS) - 1 - i
        ax.errorbar(m, y, xerr=e, fmt="o", markersize=8, color=COLOR[a],
                    ecolor=COLOR[a], elinewidth=2.0, capsize=4, zorder=3)
        ax.annotate(f"{m:.2f}", (m, y), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(AGENTS)))
    ax.set_yticklabels([a for a in reversed(AGENTS)], fontsize=9)
    for tick, a in zip(ax.get_yticklabels(), reversed(AGENTS)):
        tick.set_color(COLOR[a])
    ax.set_ylim(-0.6, len(AGENTS) - 0.4)
    ax.set_xlabel(xlabel, fontsize=8.5, color=INK2)
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=16)
    ax.annotate(subtitle, xy=(0, 1.0), xycoords="axes fraction",
                xytext=(0, 6), textcoords="offset points",
                fontsize=8, color=INK2, ha="left")


def suicide_panel(ax, classic, tourney):
    """Own-bomb deaths per round in both settings: one axis, two agents."""
    settings = [("tournament, 150 rounds", tourney), ("classic, 10 matches", classic)]
    height = 0.34
    for row, (label, data) in enumerate(settings):
        for i, a in enumerate(AGENTS):
            m, e = ci95(data[a]["suicides"])
            y = row + (0.5 - i) * (height + 0.04)
            ax.barh(y, m, height=height, color=COLOR[a], zorder=3)
            if e > 0:
                ax.errorbar(m, y, xerr=e, fmt="none", ecolor=INK2,
                            elinewidth=1.2, capsize=3, zorder=4)
            ax.annotate(f"{m:.3f}", (m + e, y), textcoords="offset points",
                        xytext=(10, -3), ha="left", fontsize=8, color=INK)
    ax.set_yticks(range(len(settings)))
    ax.set_yticklabels([s[0] for s in settings], fontsize=8.5)
    ax.set_ylim(-0.55, len(settings) - 0.45)
    ax.set_xlim(0, 0.46)
    ax.set_xlabel("own-bomb deaths per round", fontsize=8.5, color=INK2)
    ax.set_title("Cost in self-kills", fontsize=10, color=INK, loc="left", pad=16)
    ax.annotate("indistinguishable on classic; in the tournament rcp-linear's rises, tree_agent's does not",
                xy=(0, 1.0), xycoords="axes fraction", xytext=(0, 6),
                textcoords="offset points", fontsize=8, color=INK2, ha="left")


def main():
    classic, n_matches = load_classic()
    tourney, n_rounds = load_tournament()

    print(f"matched protocol -- classic, {n_matches} seeded matches of 50 rounds")
    for a in AGENTS:
        m, e = ci95(classic[a]["score"])
        print(f"  {a:<12} {m:6.2f} +- {e:.2f} points/round")
    d = np.array(classic["rcp-linear"]["score"]) - np.array(classic["tree_agent"]["score"])
    m, e = ci95(d)
    print(f"  paired difference (rcp - tree) {m:+.2f} +- {e:.2f}"
          f"   paired t p={stats.ttest_rel(classic['rcp-linear']['score'], classic['tree_agent']['score']).pvalue:.3f}")

    print(f"\ntournament -- {n_rounds} rounds, n = 1 match, recovered from the engine log")
    for a in AGENTS:
        m, e = ci95(tourney[a]["score"])
        print(f"  {a:<12} {m:6.2f} +- {e:.2f} points/round")
    d2 = tourney["tree_agent"]["score"] - tourney["rcp-linear"]["score"]
    m, e = ci95(d2)
    print(f"  paired difference (tree - rcp) {m:+.2f} +- {e:.2f}"
          f"   paired t p={stats.ttest_rel(tourney['tree_agent']['score'], tourney['rcp-linear']['score']).pvalue:.2f}")
    st, sr = tourney["tree_agent"]["suicides"], tourney["rcp-linear"]["suicides"]
    print(f"  own-bomb deaths/round  tree {st.mean():.3f}  rcp {sr.mean():.3f}"
          f"   paired t p={stats.ttest_rel(st, sr).pvalue:.4f}")

    fig = plt.figure(figsize=(7.6, 5.1))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.95, wspace=0.28)
    ax_classic = fig.add_subplot(gs[0, 0])
    ax_tourney = fig.add_subplot(gs[0, 1])
    ax_suicide = fig.add_subplot(gs[1, :])

    dot_panel(ax_classic, {a: classic[a]["score"] for a in AGENTS},
              "Matched protocol",
              f"classic, {n_matches} seeded matches of 50 rounds",
              "points per round")
    dot_panel(ax_tourney, {a: tourney[a]["score"] for a in AGENTS},
              "The one recorded tournament",
              f"coin-rich board, {n_rounds} rounds, $n=1$",
              "points per round")
    suicide_panel(ax_suicide, classic, tourney)

    for ax in (ax_classic, ax_tourney, ax_suicide):
        style(ax)

    handles = [plt.Line2D([], [], color=COLOR[a], marker="o", markersize=7,
                          linewidth=2.0, label=a) for a in AGENTS]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.03))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "head_to_head.png"
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
