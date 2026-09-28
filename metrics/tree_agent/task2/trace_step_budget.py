import argparse
import importlib
import os
import pickle
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import settings as s  # noqa: E402
from environment import BombeRLeWorld, WorldArgs  # noqa: E402


def make_world(agent_name, seed, scenario):
    args = WorldArgs(
        no_gui=True, fps=15, turn_based=False, update_interval=0.1, save_replay=False, replay=None,
        make_video=False, continue_without_training=True, log_dir=os.path.join(ROOT, "logs"),
        save_stats=False, match_name=None, seed=seed, silence_errors=False, scenario=scenario,
    )
    return BombeRLeWorld(args, [(agent_name, False)])


def greedy_action(features, callbacks, model, state):
    if state["in_escape_window"] and state["escape_direction"] is not None:
        return state["escape_direction"]
    values = features.q_values(model, state)
    if hasattr(callbacks, "executable_action_mask"):
        mask = callbacks.executable_action_mask(state)
    else:  # pre-revision callbacks: mask needs a `self` carrying the bomb history
        agent_stub = type("Stub", (), {"opponent_has_bombed": {}})()
        mask = callbacks._safety_mask(agent_stub, state)
        if not mask.any():
            mask = np.array([features.state_action_features(state, a)[1] == 1.0 for a in features.ACTIONS])
    return features.ACTIONS[int(np.argmax(np.where(mask, values, -np.inf)))]


def navigation_target_kind(state):
    if state.get("coin_distance_map") is not None:
        return "coin"
    if state.get("crate_distance_map") is not None:
        return "crate"
    return None


def crates_pending(state, field):
    pending = set()
    for blast in state["bomb_blasts"].values():
        for tile in blast:
            if field[tile] == 1:
                pending.add(tile)
    return pending


def target_tile_is_pending(state, field, pending):
    dist_map = state["crate_distance_map"]
    if dist_map is None or not pending:
        return False
    pos = state["position"]
    if not np.isfinite(dist_map[pos]):
        return False
    # walk the gradient down to a distance-0 tile
    cur = pos
    for _ in range(int(dist_map[pos]) + 1):
        if dist_map[cur] == 0:
            break
        for dx, dy in features_directions():
            nxt = (cur[0] + dx, cur[1] + dy)
            if 0 <= nxt[0] < field.shape[0] and 0 <= nxt[1] < field.shape[1] and dist_map[nxt] == dist_map[cur] - 1:
                cur = nxt
                break
    adjacent_crates = {
        (cur[0] + dx, cur[1] + dy) for dx, dy in features_directions()
        if 0 <= cur[0] + dx < field.shape[0] and 0 <= cur[1] + dy < field.shape[1] and field[cur[0] + dx, cur[1] + dy] == 1
    }
    return bool(adjacent_crates) and adjacent_crates <= pending


def features_directions():
    return [(0, -1), (0, 1), (-1, 0), (1, 0)]


def run(agent_name, rounds, seed, scenario, verbose, json_out=None):
    features = importlib.import_module(f"agent_code.{agent_name}.features")
    callbacks = importlib.import_module(f"agent_code.{agent_name}.callbacks")
    model_path = os.path.join(ROOT, "agent_code", agent_name, "tree-model.pt")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    world = make_world(agent_name, seed, scenario)
    agent = world.agents[0]
    totals = Counter()
    per_round = []
    cycles = []
    yields = []
    stuck_rounds = 0

    for r in range(rounds):
        world.new_round()
        world.user_input = None
        classes = Counter()
        last_drop_step = None
        coins = 0
        crates_before = int((world.arena == 1).sum())
        last_progress_step = 0
        stuck = False
        pending_bombs = []  # (drop_step, blast tiles, crates in blast at drop)
        while world.running:
            gs = world.get_state_for_agent(agent)
            if gs is None:
                break
            field = gs["field"]
            state = features.state_features(gs, use_target_aware_escape=True, opponent_has_bombed={})
            action = greedy_action(features, callbacks, model, state)
            phi = features.state_action_features(state, action)
            pending = crates_pending(state, field)
            kind = navigation_target_kind(state)

            if phi[1] == 0.0:
                cls = "invalid"
            elif state["in_escape_window"]:
                cls = "escape"
            elif action == "WAIT":
                cls = "wait"
            elif action == "BOMB":
                cls = "bomb"
            elif kind == "crate" and target_tile_is_pending(state, field, pending) and phi[8] == 1.0:
                cls = "toward_pending"
            elif kind == "coin":
                cls = "coin_chase"
            elif kind == "crate":
                cls = "crate_walk"
            else:
                cls = "other_move"
            classes[cls] += 1

            if action == "BOMB" and phi[1] == 1.0:
                step = world.step + 1
                if last_drop_step is not None:
                    cycles.append(step - last_drop_step)
                last_drop_step = step
                yields.append(int(phi[6]))
                last_progress_step = step

            score_before = agent.score
            world.do_step()
            if agent.score > score_before:
                coins += agent.score - score_before
                last_progress_step = world.step
            if world.step - last_progress_step >= 50 and not stuck:
                stuck = True
        crates_after = int((world.arena == 1).sum())
        if stuck:
            stuck_rounds += 1
        per_round.append((coins, crates_before - crates_after, classes["bomb"], world.step, stuck))
        totals.update(classes)
        if verbose:
            print(f"round {r+1:3d}: coins {coins:2d} crates {crates_before - crates_after:3d} bombs {classes['bomb']:2d} "
                  f"steps {world.step:3d} {'STUCK' if stuck else ''}")

    n = len(per_round)
    steps_total = sum(p[3] for p in per_round)
    print(f"\n== {agent_name} on {scenario}, {n} greedy rounds (seed {seed}) ==")
    print(f"coins/round        {sum(p[0] for p in per_round)/n:6.2f}")
    print(f"crates/round       {sum(p[1] for p in per_round)/n:6.2f}")
    print(f"bombs/round        {sum(p[2] for p in per_round)/n:6.2f}")
    print(f"crates/bomb        {sum(p[1] for p in per_round)/max(1, sum(p[2] for p in per_round)):6.2f}")
    print(f"steps/bomb cycle   {np.mean(cycles) if cycles else float('nan'):6.2f}   (median {np.median(cycles) if cycles else float('nan'):.0f}, floor ~7)")
    print(f"full clears        {sum(1 for p in per_round if p[0] >= 50)}/{n}")
    print(f"stuck rounds       {stuck_rounds}/{n}   (>=50 steps without a coin or a bomb)")
    print("step budget (share of all steps):")
    classes_order = ["crate_walk", "coin_chase", "escape", "toward_pending", "bomb", "wait", "invalid", "other_move"]
    for cls in classes_order:
        print(f"  {cls:15s} {totals[cls]/steps_total:6.1%}  ({totals[cls]/n:5.1f} steps/round)")

    if json_out:
        import json
        summary = {
            "agent": agent_name, "scenario": scenario, "seed": seed, "rounds": n,
            "coins_per_round": sum(p[0] for p in per_round) / n,
            "crates_per_round": sum(p[1] for p in per_round) / n,
            "bombs_per_round": sum(p[2] for p in per_round) / n,
            "steps_per_bomb_cycle_mean": float(np.mean(cycles)) if cycles else None,
            "steps_per_bomb_cycle_median": float(np.median(cycles)) if cycles else None,
            "bomb_cycles": [int(c) for c in cycles],
            "full_clears": sum(1 for p in per_round if p[0] >= 50),
            "stuck_rounds": stuck_rounds,
            "steps_per_round_by_class": {cls: totals[cls] / n for cls in classes_order},
            "per_round": [{"coins": p[0], "crates": p[1], "bombs": p[2], "steps": p[3], "stuck": p[4]} for p in per_round],
        }
        with open(json_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {json_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", default="tree_agent")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", default="loot-crate")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--json", default=None, help="also write the aggregated numbers to this path")
    a = parser.parse_args()
    run(a.agent, a.rounds, a.seed, a.scenario, a.verbose, a.json)

