import argparse
import importlib
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import events as e  # noqa: E402
from environment import BombeRLeWorld, WorldArgs  # noqa: E402


def run(agent_name, opponent, rounds, seed, verbose_deaths):
    F = importlib.import_module(f"agent_code.{agent_name}.features")
    args = WorldArgs(no_gui=True, fps=15, turn_based=False, update_interval=0.1, save_replay=False, replay=None,
                     make_video=False, continue_without_training=True, log_dir=os.path.join(ROOT, "logs"),
                     save_stats=False, match_name=None, seed=seed, silence_errors=False, scenario="classic")
    world = BombeRLeWorld(args, [(agent_name, False), (opponent, False)])
    me, opp = world.agents[0], world.agents[1]
    causes = Counter()
    deaths = kills = 0
    details = []
    for r in range(rounds):
        world.new_round()
        world.user_input = None
        hist = []
        while world.running:
            gs = world.get_state_for_agent(me)
            if gs is None:
                break
            st = F.state_features(gs, use_target_aware_escape=True, opponent_has_bombed={})
            hist.append(dict(
                step=world.step + 1, pos=(int(gs["self"][3][0]), int(gs["self"][3][1])),
                opp=None if opp.dead else (int(opp.x), int(opp.y)),
                esc=st["escape_direction"], win=st["in_escape_window"],
                own=[(int(b.x), int(b.y), b.timer) for b in world.bombs if b.owner is me],
                oppb=[(int(b.x), int(b.y), b.timer) for b in world.bombs if b.owner is not me],
            ))
            world.do_step()
            hist[-1]["action"] = me.last_action
            if e.KILLED_OPPONENT in me.events:
                kills += 1
            if me.dead:
                deaths += 1
                killed_self = e.KILLED_SELF in me.events
                recent = hist[-5:]
                no_route = any(h["win"] and h["esc"] is None for h in recent)
                opp_bomb_recent = any(h["oppb"] for h in recent)
                if not killed_self:
                    cause = "killed by opponent bomb"
                elif no_route and opp_bomb_recent:
                    cause = "own bomb: escape route lost after opponent bomb"
                elif no_route:
                    cause = "own bomb: no escape route found"
                elif any(h["win"] and h["esc"] is not None and h["action"] != h["esc"] for h in recent):
                    cause = "own bomb: did not follow escape direction"
                else:
                    cause = "own bomb: followed route but still died"
                causes[cause] += 1
                if killed_self and len(details) < verbose_deaths:
                    details.append((r + 1, cause, hist[-6:]))
                break
    print(f"{agent_name} vs {opponent}: {rounds} rounds, our deaths {deaths}, our kills {kills}")
    for cause, n in causes.most_common():
        print(f"  {n:3d}  {cause}")
    for r, cause, steps in details:
        print(f"\nround {r}: {cause}")
        for h in steps:
            print(f"   step {h['step']:3d} pos {h['pos']} act {str(h['action']):5s} esc {str(h['esc']):5s} "
                  f"window {str(h['win']):5s} opp {h['opp']} opp bombs {h['oppb']} own bombs {h['own']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", default="tree_agent")
    p.add_argument("--opponent", default="rule_based_agent")
    p.add_argument("--rounds", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose-deaths", type=int, default=4, help="how many own-bomb deaths to print step by step")
    a = p.parse_args()
    run(a.agent, a.opponent, a.rounds, a.seed, a.verbose_deaths)
