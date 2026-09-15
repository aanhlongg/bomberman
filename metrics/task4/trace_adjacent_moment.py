"""
Find the exact step(s) where linear_agent is orthogonally adjacent
(Manhattan distance 1) to coin_collector_agent, and dump the full
state_action_features(state, "BOMB") vector plus Q-values for every
action at that moment, to see exactly why BOMB isn't chosen.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import numpy as np
from environment import BombeRLeWorld, WorldArgs
from agent_code.linear_agent.features import state_features, state_action_features, q_values, ACTIONS

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trace_logs")
os.makedirs(LOG_DIR, exist_ok=True)

with open("agent_code/linear_agent/linear-model.pt", "rb") as f:
    weights = pickle.load(f)

args = WorldArgs(
    no_gui=True, fps=15, turn_based=False, update_interval=0.1,
    save_replay=False, replay=None, make_video=False,
    continue_without_training=True, log_dir=LOG_DIR, save_stats=False,
    match_name=None, seed=None, silence_errors=False, scenario="classic",
)

world = BombeRLeWorld(args, [("linear_agent", False), ("coin_collector_agent", False)])


def find_agent(world, name):
    for a in world.agents:
        if a.name == name:
            return a
    raise KeyError(name)


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


WANT_MOMENTS = 6
found = 0
round_num = 0
MAX_ROUNDS = 30

while found < WANT_MOMENTS and round_num < MAX_ROUNDS:
    round_num += 1
    world.new_round()
    la = find_agent(world, "linear_agent")
    cc = find_agent(world, "coin_collector_agent")

    while world.running and found < WANT_MOMENTS:
        was_dead_before = la.dead or cc.dead
        was_adjacent_before = (
            not was_dead_before and manhattan((la.x, la.y), (cc.x, cc.y)) == 1
        )
        world.do_step()
        gs = la.last_game_state if not was_dead_before else None
        if was_adjacent_before and gs is not None:
            state = state_features(gs, use_target_aware_escape=True)
            bomb_feat = state_action_features(state, "BOMB")
            qs = q_values(weights, state)
            action_taken = world.replay["actions"]["linear_agent"][-1]
            print(f"round={round_num} step={gs['step']} la_pos={gs['self'][3]} others={[o[3] for o in gs['others']]}")
            print(f"  bomb_feat: valid={bomb_feat[1]} escape_exists={bomb_feat[5]} crate_hit={bomb_feat[6]} bomb_hits_opp={bomb_feat[10]}")
            print(f"  in_escape_window={state['in_escape_window']} bomb_available={state['bomb_available']}")
            print(f"  Q-values: {dict(zip(ACTIONS, [round(v,2) for v in qs]))}")
            print(f"  action_taken={action_taken}")
            print()
            found += 1

world.end()
print(f"Found {found} adjacent moments in {round_num} rounds")
