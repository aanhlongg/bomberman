"""
Find moments where the board is fully cleared (no coins, no reachable
crates) vs peaceful_agent, and dump exactly what the agent does: does it
approach the opponent, or does it WAIT? Shows the BFS distance to the
nearest opponent, Q-values for every action, and the action actually taken.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pickle
from environment import BombeRLeWorld, WorldArgs
from agent_code.linear_agent.features import state_features, q_values, ACTIONS

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

world = BombeRLeWorld(args, [("linear_agent", False), ("peaceful_agent", False)])


def find_agent(world, name):
    for a in world.agents:
        if a.name == name:
            return a
    raise KeyError(name)


def board_cleared(state):
    if state["coin_distance_map"] is not None:
        return False
    if state["crate_distance_map"] is None:
        return False
    return not np.any(np.isfinite(state["crate_distance_map"]))


WANT_SAMPLES = 15
MAX_ROUNDS = 30
found = 0
round_num = 0
samples_this_round_cap = 3  # avoid one long round dominating the sample

while found < WANT_SAMPLES and round_num < MAX_ROUNDS:
    round_num += 1
    world.new_round()
    la = find_agent(world, "linear_agent")
    pc = find_agent(world, "peaceful_agent")
    samples_this_round = 0

    while world.running and found < WANT_SAMPLES:
        was_dead = la.dead or pc.dead
        world.do_step()
        gs = la.last_game_state if not was_dead else None
        if gs is None or samples_this_round >= samples_this_round_cap:
            continue
        state = state_features(gs, use_target_aware_escape=True)
        if state is None or not board_cleared(state):
            continue
        action_taken = world.replay["actions"]["linear_agent"][-1]
        qs = q_values(weights, state)
        opp_dist = state["opponent_distance_map"][state["position"]] if state["opponent_distance_map"] is not None else None
        print(f"round={round_num} step={gs['step']} la_pos={gs['self'][3]} "
              f"pc_pos={[o[3] for o in gs['others']]} bomb_available={gs['self'][2]}")
        print(f"  opponent_distance={opp_dist} in_escape_window={state['in_escape_window']}")
        print(f"  Q-values: {dict(zip(ACTIONS, [round(v,2) for v in qs]))}")
        print(f"  action_taken={action_taken}")
        print()
        found += 1
        samples_this_round += 1

world.end()
print(f"Found {found} cleared-board samples in {round_num} rounds")
