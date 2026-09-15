"""
Trace linear_agent's deaths (both self-kills and opponent-kills) vs
rule_based_agent, using the current (freshly-trained-on-this-matchup)
shipped model, to find the dominant failure mode.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import events as e
from environment import BombeRLeWorld, WorldArgs
from agent_code.linear_agent.features import state_features, state_action_features, ACTIONS

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trace_logs")
os.makedirs(LOG_DIR, exist_ok=True)

args = WorldArgs(
    no_gui=True, fps=15, turn_based=False, update_interval=0.1,
    save_replay=False, replay=None, make_video=False,
    continue_without_training=True, log_dir=LOG_DIR, save_stats=False,
    match_name=None, seed=None, silence_errors=False, scenario="classic",
)

world = BombeRLeWorld(args, [("linear_agent", False), ("rule_based_agent", False)])

BUFFER_LEN = 12
MAX_ROUNDS = 40
WANT_SELF_KILL = 4
WANT_OPPONENT_KILL = 4

self_kill_traces = []
opponent_kill_traces = []

out_path = os.path.join(LOG_DIR, "death_traces_rulebased.txt")
out_f = open(out_path, "w")


def find_agent(world, name):
    for a in world.agents:
        if a.name == name:
            return a
    raise KeyError(name)


def fmt_bombs(bombs):
    return [(tuple(int(v) for v in pos), int(t)) for pos, t in bombs]


def dump_trace(out_f, round_num, cause, buffer):
    out_f.write(f"\n{'='*70}\nROUND {round_num} -- linear_agent died: {cause}\n{'='*70}\n")
    for entry in buffer:
        step, gs, action, events_this_step, opp_action = entry
        state = state_features(gs, use_target_aware_escape=True)
        pos = gs["self"][3]
        others = [(o[0], o[3]) for o in gs["others"]]
        bombs = fmt_bombs(gs["bombs"])
        explosion_cells = [tuple(int(v) for v in xy) for xy in zip(*np.nonzero(gs["explosion_map"] > 0))]
        in_escape = state["in_escape_window"] if state else None
        escape_dir = state["escape_direction"] if state else None
        taken_feat = state_action_features(state, action) if state else None
        moves_into_danger = taken_feat[4] if taken_feat is not None else None
        escape_correct = taken_feat[7] if taken_feat is not None else None
        out_f.write(
            f"step={step:3d} pos={pos} others={others} bombs={bombs} "
            f"explosions={explosion_cells}\n"
            f"          action_taken={action:6s} opponent_action={opp_action} events={events_this_step}\n"
            f"          in_escape_window={in_escape} escape_direction={escape_dir} "
            f"moves_into_avoidable_danger={moves_into_danger} escape_correct_move={escape_correct}\n"
        )
    out_f.write("\n")
    out_f.flush()


round_num = 0
while (len(self_kill_traces) < WANT_SELF_KILL or len(opponent_kill_traces) < WANT_OPPONENT_KILL) and round_num < MAX_ROUNDS:
    round_num += 1
    world.new_round()
    la = find_agent(world, "linear_agent")
    buffer = []
    captured_this_round = False

    while world.running:
        was_dead_before = la.dead
        world.do_step()
        pre_state = la.last_game_state if not was_dead_before else None
        action = world.replay["actions"]["linear_agent"][-1] if world.replay["actions"]["linear_agent"] else None
        opp_actions = world.replay["actions"].get("rule_based_agent")
        opp_action = opp_actions[-1] if opp_actions else None
        events_this_step = list(la.events) if (not was_dead_before) else []
        if pre_state is not None:
            buffer.append((world.step, pre_state, action, events_this_step, opp_action))
            if len(buffer) > BUFFER_LEN:
                buffer.pop(0)

        if not captured_this_round:
            is_self_kill = e.KILLED_SELF in events_this_step
            is_pure_opponent_kill = e.GOT_KILLED in events_this_step and e.KILLED_SELF not in events_this_step
            if is_self_kill and len(self_kill_traces) < WANT_SELF_KILL:
                dump_trace(out_f, round_num, "KILLED_SELF" + (" + KILLED_OPPONENT (mutual)" if e.KILLED_OPPONENT in events_this_step else ""), buffer)
                self_kill_traces.append(round_num)
                captured_this_round = True
            elif is_pure_opponent_kill and len(opponent_kill_traces) < WANT_OPPONENT_KILL:
                dump_trace(out_f, round_num, "PURE opponent-bomb kill (no KILLED_SELF)", buffer)
                opponent_kill_traces.append(round_num)
                captured_this_round = True

world.end()
out_f.close()
print("SELF_KILL traces from rounds:", self_kill_traces)
print("OPPONENT_KILL traces from rounds:", opponent_kill_traces)
print("total rounds simulated:", round_num)
print("written to:", out_path)
