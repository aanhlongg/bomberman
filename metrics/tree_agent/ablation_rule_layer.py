import argparse
import importlib
import inspect
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import events as e  # noqa: E402
from environment import BombeRLeWorld, WorldArgs  # noqa: E402

ARMS = ["shipped", "no_escape", "q_zero", "no_mask", "veto_radius_1"]
OUT_DIR = REPO_ROOT / "results" / "tree_agent" / "ablation_rule_layer"

# the exact branch condition in callbacks.act that hands control to the BFS route
ESCAPE_BRANCH = 'elif state is not None and state["in_escape_window"] and state["escape_direction"] is not None:'


def apply_arm(callbacks, features, arm):

    if arm == "no_escape":
        src = inspect.getsource(callbacks.act)
        if ESCAPE_BRANCH not in src:
            raise RuntimeError(
                "callbacks.act no longer contains the expected escape branch; "
                "the ablation would silently be a no-op. Update ESCAPE_BRANCH."
            )
        src = src.replace(ESCAPE_BRANCH, "elif False:")
        namespace = dict(callbacks.__dict__)
        exec(compile(src, "<ablation:no_escape>", "exec"), namespace)
        callbacks.act = namespace["act"]

    elif arm == "q_zero":
        n = len(features.ACTIONS)
        callbacks.q_values = lambda _model, _state: np.zeros(n)

    elif arm == "veto_radius_1":

        callbacks.VETO_RADIUS_ACTIVE = 1

    elif arm == "no_mask":
        def validity_only(_self, state):
            return np.array(
                [features.state_action_features(state, a)[features.F_VALID] == 1.0
                 for a in features.ACTIONS]
            )
        callbacks._safety_mask = validity_only


def fresh_agent_modules(arm):

    for name in ("agent_code.tree_agent.callbacks", "agent_code.tree_agent.features"):
        sys.modules.pop(name, None)
    features = importlib.import_module("agent_code.tree_agent.features")
    callbacks = importlib.import_module("agent_code.tree_agent.callbacks")
    # keep the per-step metrics CSV out of it; it is opened "w" and we are not measuring it
    callbacks._log_step = lambda *a, **k: None
    apply_arm(callbacks, features, arm)
    return callbacks, features


def run_arm(arm, rounds, seed, opponent, scenario="classic"):
  
    callbacks, features = fresh_agent_modules(arm)
    args = WorldArgs(
        no_gui=True, fps=15, turn_based=False, update_interval=0.1, save_replay=False,
        replay=None, make_video=False, continue_without_training=True,
        log_dir=str(REPO_ROOT / "logs"), save_stats=False, match_name=None,
        seed=seed, silence_errors=False, scenario=scenario,
    )
    roster = [("tree_agent", False)]
    if opponent != "none":
        roster.append((opponent, False))
    world = BombeRLeWorld(args, roster)
    me = world.agents[0]
    opp = world.agents[1] if opponent != "none" else None

    diag = Counter()
    per_round = []
    for _ in range(rounds):
        world.new_round()
        world.user_input = None
        while world.running:
            gs = world.get_state_for_agent(me)
            if gs is not None and not me.dead:

                ohb = getattr(me.backend.runner.fake_self, "opponent_has_bombed", {})
                st = features.state_features(gs, use_target_aware_escape=True,
                                             opponent_has_bombed=ohb)
                diag["steps"] += 1
                if st is not None and st["in_escape_window"]:
                    diag["in_escape_window"] += 1
                    if st["escape_direction"] is not None:
                        diag["escape_route_found"] += 1
                        if features.state_action_features(st, st["escape_direction"])[features.F_VALID] != 1.0:
                            diag["escape_route_illegal"] += 1
            world.do_step()
        per_round.append({
            "score": me.score,
            "opp_score": opp.score if opp is not None else 0,
            "coins": me.statistics["coins"],
            "kills": me.statistics["kills"],
            "suicides": me.statistics["suicides"],
            "invalid": me.statistics["invalid"],
            "steps": world.step,
        })
    totals = {
        "score": me.total_score,
        "opp_score": opp.total_score if opp is not None else 0,
        **{k: me.lifetime_statistics[k] for k in ("coins", "kills", "suicides", "invalid", "bombs", "moves")},
        "opp_suicides": opp.lifetime_statistics["suicides"] if opp is not None else 0,
    }
    return totals, per_round, dict(diag)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--arms", nargs="*", default=ARMS, choices=ARMS)
    ap.add_argument("--opponent", default="rule_based_agent",
                    help='"none" runs solo, the only fully reproducible setting')
    ap.add_argument("--scenario", default="classic")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed_base + i for i in range(args.seeds)]
    print(f"{len(args.arms)} arms x {len(seeds)} seeds x {args.rounds} rounds vs {args.opponent}")

    for arm in args.arms:
        runs = []
        for seed in seeds:
            t0 = time.time()
            totals, per_round, diag = run_arm(arm, args.rounds, seed, args.opponent, args.scenario)
            runs.append({"seed": seed, "totals": totals, "per_round": per_round, "diagnostics": diag})
            esc = diag.get("escape_route_found", 0) / max(diag.get("steps", 1), 1)
            print(f"  [{arm:<9}] seed {seed}: {totals['score']:>5} vs {totals['opp_score']:<5} "
                  f"suic {totals['suicides']:>3}  invalid {totals['invalid']:>4}  "
                  f"escape-window {esc:.1%}  ({time.time()-t0:.0f}s)", flush=True)
        (OUT_DIR / f"{arm}.json").write_text(json.dumps({
            "arm": arm, "rounds_per_match": args.rounds, "opponent": args.opponent,
            "seeds": seeds, "runs": runs,
        }, indent=2))
    print(f"\n-> {OUT_DIR}")


if __name__ == "__main__":
    main()
