"""
Diagnose stalemate rounds (400 steps, no kill, no suicide) vs
coin_collector_agent using the current shipped model. Samples the
distance-to-opponent trajectory every 20 steps rather than dumping full
per-step detail, plus summary stats (closest approach, bomb count, whether
a bomb was ever dropped while adjacent to the opponent).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from environment import BombeRLeWorld, WorldArgs
from agent_code.linear_agent.features import state_features

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trace_logs")
os.makedirs(LOG_DIR, exist_ok=True)

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


WANT_STALEMATES = 5
MAX_ROUNDS = 40
found = 0
round_num = 0

out_path = os.path.join(LOG_DIR, "stalemate_traces.txt")
out_f = open(out_path, "w")

while found < WANT_STALEMATES and round_num < MAX_ROUNDS:
    round_num += 1
    world.new_round()
    la = find_agent(world, "linear_agent")
    cc = find_agent(world, "coin_collector_agent")

    samples = []
    bombs_dropped = 0
    bombs_while_adjacent = 0
    min_distance = 999
    last_action_la = None

    while world.running:
        was_dead = la.dead or cc.dead
        world.do_step()
        action = world.replay["actions"]["linear_agent"][-1] if world.replay["actions"]["linear_agent"] else None
        if action == "BOMB":
            bombs_dropped += 1
        if not la.dead and not cc.dead:
            dist = manhattan((la.x, la.y), (cc.x, cc.y))
            min_distance = min(min_distance, dist)
            if action == "BOMB" and dist <= 1:
                bombs_while_adjacent += 1
            if world.step % 20 == 0:
                samples.append((world.step, (la.x, la.y), (cc.x, cc.y), dist))

    is_stalemate = (not la.dead) and (not cc.dead) and world.step >= 400
    if is_stalemate:
        found += 1
        out_f.write(f"\n{'='*70}\nROUND {round_num} -- STALEMATE (steps={world.step})\n{'='*70}\n")
        out_f.write(f"min_distance_reached={min_distance} bombs_dropped={bombs_dropped} bombs_while_adjacent={bombs_while_adjacent}\n")
        for step, la_pos, cc_pos, dist in samples:
            out_f.write(f"  step={step:3d} la_pos={la_pos} cc_pos={cc_pos} distance={dist}\n")
        out_f.flush()

world.end()
out_f.close()
print(f"Found {found} stalemates in {round_num} rounds")
print("written to:", out_path)
