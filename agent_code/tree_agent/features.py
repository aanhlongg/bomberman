"""
State representation and the Q-function for the tree_agent.

Unlike linear_agent (Q(s,a) = weights . phi(s,a)), the tree_agent is a
tree-ensemble function approximator: Q(s,a) = model.predict(phi(s,a)), where
model is a single sklearn regressor (ExtraTreesRegressor, see train.py)
shared across all six actions -- exactly like linear_agent, the action isn't
a separate categorical input; it's encoded implicitly by evaluating
phi(s, .) once per candidate action.

Task 2 scope (final_project.pdf, section 4.2): "On a game board with crates,
collect coins as before while safely blowing up crates that are in the way."
This file ports linear_agent's own (architecture-independent) Task 2 feature
set and ground-truth BFS logic almost verbatim -- danger detection, escape
routing, crate-hit checking are all deterministic game-state computations
that don't depend on the function approximator. What's deliberately NOT
ported is linear_agent's "synthetic counterfactual update" (a direct
weight-space nudge specific to per-step SGD on a linear model) -- there is
no equivalent operation for a batch-refit tree ensemble. Whether the
collinearity problem that update was built to fix (see linear_agent's Task 2
report chapter) even affects a tree ensemble the same way is an open
empirical question this task's own training run will answer: a tree does
not need to attribute credit between two features that always co-occur the
way a linear model's weights do, so the hypothesis going in is that it
won't need the workaround -- but this is being tested, not assumed.

Task 3 scope (final_project.pdf, section 4.2): "hunt and blow up
peaceful_agent (easy) and coin_collector_agent (hard)." Adds
moves_to_opponent and bomb_hits_opponent_if_bomb, ported from linear_agent's
own Task 3 feature set -- again deterministic BFS/blast-simulation logic,
not architecture-specific. The synthetic counterfactual update for
escape_correct_move is deliberately NOT ported (same SGD-specific reasoning
as above -- there is no per-sample weight to nudge here). Task 3's own
opponents (peaceful_agent, coin_collector_agent) never bomb an adjacent
opponent deliberately, so retaliation-aware escape logic was deliberately
deferred to Task 4 at that stage -- opponents' CURRENT positions were
already treated as forbidden (occupied) tiles during escape planning,
since an opponent standing on a tile blocks movement through it this
instant regardless of whether it might retaliate.

Task 4 scope (final_project.pdf, section 4): "hold your own against one or
more opposing agents (e.g. the full-strength rule_based_agent...)." Adds the
retaliation-aware extension to _escape_exists_after_bomb deferred above
(accounting for an adjacent, bomb-capable opponent bombing back), ported
from linear_agent unchanged -- again deterministic blast-simulation logic.
linear_agent's own investigation found this feature alone insufficient
(self-kill rate barely moved); the fix that actually worked was a
non-learned, unconditional veto (callbacks.py's
_adjacent_opponent_can_bomb) applied at action-selection time, not a
feature/reward change at all -- ported there, not here.

The escape-routing tie-break refinement (target-aware tie-breaking among
multiple equally-short escape routes, linear_agent's later "Routing
Efficiency" fix) is also deliberately NOT ported yet -- linear_agent
discovered that fix only after escape safety and the bombing decision were
already solved, and revisiting efficiency before correctness is settled
would be jumping ahead of the same order of discovery.

Public API used by callbacks.py / train.py:
    state_features(game_state)            -- precompute the state-level parts
                                              shared by every action this step
    state_action_features(state, action)  -- the feature vector for one action
    q_values(model, state)                -- Q(s, .) for every action (model
                                              may be None before the first
                                              Fitted-Q-Iteration refit, see
                                              train.py -- returns all zeros)
    coin_potential / crate_potential /
    escape_potential(state)               -- potential-based reward-shaping
                                              terms, summed in train.py
    N_FEATURES, GAMMA, ESCAPE_BUDGET,
    TARGET_UNREACHABLE                     -- shared constants

Everything else is a private helper (leading underscore).
"""

from collections import deque

import numpy as np

import settings as s

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]

# image coordinates, top left is (0,0)
DIRECTIONS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

# feature vector layout (index: name) -- deliberately kept in the same order
# as linear_agent's own layout, so the two agents' feature tables read the
# same way in the report even though the underlying model differs.
# 0: moves_to_coin              6: crate_count_if_bomb
# 1: valid                      7: escape_correct_move
# 2: is_wait                    8: moves_to_crate
# 3: is_bomb                    9: moves_to_opponent
# 4: moves_into_avoidable_danger    10: bomb_hits_opponent_if_bomb
# 5: escape_exists_if_bomb
#
# Routing-efficiency block (appended so 0-10 keep their meaning for
# callbacks.py / train.py's named lookups; see _select_target for the
# target these refer to):
# 11: target_distance_after     14: bomb_cooldown
# 12: target_distance_delta     15: target_is_coin
# 13: yield_at_landing_tile
#
# 11: BFS distance from the tile this action lands on (current tile for
#     WAIT / BOMB / an invalid move) to the selected navigation target;
#     TARGET_UNREACHABLE when there is no target or no path.
# 12: (11) minus the same distance at the current tile: -1 toward, +1 away,
#     0 for WAIT / BOMB / invalid (a graded, signed version of 0 / 8, which
#     only say "toward").
# 13: crates a bomb dropped on the landing tile would destroy, not counting
#     crates already inside an active blast -- lets a MOVE see that it is
#     stepping onto (or off) a 4-crate spot, where 6 only informs BOMB.
# 14: steps until bomb_available comes back (0 if it is available now) --
#     lets the model pre-walk to the next spot during its own cooldown
#     instead of standing still. See _bomb_cooldown for the engine timing.
# 15: 1.0 if the selected target is a coin, 0.0 if it is a bomb spot (or
#     there is none), so a tree can condition 11-13 on the target kind.
#
# 4 and 7 are mutually exclusive by construction (gated on in_escape_window,
# see state_features): each is forced to 0 whenever the other's context is
# active. 0 and 8 are likewise mutually exclusive by construction (gated on
# which kind of target _select_target picked): moves_to_coin credits steps
# toward a coin target, moves_to_crate steps toward a bomb-spot target.
#
# 6 (crate_count_if_bomb) is a raw COUNT (how many crates this bomb would
# destroy), not linear_agent's binary "does it hit at least one" -- a
# deliberate departure, added after diagnosing that tree_agent's Task 2
# coins/round gap versus linear_agent was purely a step-budget problem (a
# generously extended step budget gets a perfect 50/50 on every test round),
# traced to escape-window overhead per bomb dominating the step count.
# Fewer, higher-yield bombs directly cuts that overhead. A binary feature
# gives the model no way to prefer a 3-crate bomb over a 1-crate one; a
# count does. linear_agent never tried this graded version for its own
# collinear reasons (see report2_body.typ's "graded escape-margin feature"
# -- a similar graded idea reverted there specifically because it was
# collinear with an existing binary feature for a LINEAR model); no such
# identifiability problem applies to a tree ensemble, so there is no
# analogous binary crate_hit feature kept alongside it here to be collinear
# with -- crate_count_if_bomb >= 1.0 fully replaces it.
#
# 9 (moves_to_opponent, Task 3) is deliberately NOT mutually exclusive with
# 0/8, and NOT gated to 0 during in_escape_window, mirroring linear_agent's
# own choice exactly: hunting is an independent objective layered on top of
# coin/crate seeking, not a replacement for it, and linear_agent found
# (twice, by reverting the gated version both times) that removing it during
# an escape window left escape_correct_move as the only action-varying
# feature there, collapsing it into a structural argmax tie rather than
# letting it develop real signal. Ported as-is without re-testing the gated
# version here, on the strength of that already-diagnosed mechanism (a
# feature-space property, not a linear-model-specific one).
N_FEATURES = 16
GAMMA = 0.95  # discount factor

# --- navigation target selection (see _select_target) ---
#
# The pre-routing agent had a hard rule: any visible coin -> chase the
# nearest coin; otherwise -> walk to the nearest crate-adjacent tile. A
# step-budget trace of the validated Task 2 model (metrics/tree_agent/task2/
# trace_step_budget.py: 44.4 coins/round, 34.5 bombs/round, 0/30 full
# clears) found where that costs steps: while chasing a coin the agent was
# standing on a >=2-crate bomb spot on 32% of steps (>=3 crates: 21%) and
# walked off it to fetch the coin first, then came back; and it spent ~18
# steps/round WAITing through its own bomb cooldown instead of walking to
# the next spot. Both are ordering/targeting decisions, so both are fixed at
# the target-selection level rather than by asking the tree ensemble to
# learn them from a 50-round replay window. Every candidate target -- each
# visible coin and each tile a bomb would clear >= 1 crate from -- gets one
# score and the best one becomes THE target for moves_to_coin /
# moves_to_crate, the potentials and the escape tie-break:
#
#   coin:  COIN_TARGET_VALUE - TARGET_DISTANCE_WEIGHT * distance
#   spot:  yield * CRATE_TARGET_VALUE - TARGET_DISTANCE_WEIGHT * distance
#          - BOMB_CYCLE_COST
#
# With the defaults a 3-crate spot under the agent's feet (3*1.5 - 1 = 3.5)
# beats a coin 2 steps away (3 - 1 = 2), a coin next door (2.5) beats a
# 1-crate spot anywhere, and a 4-crate spot beats a coin unless the coin is
# ~4 steps closer. Crates already inside an active blast count as gone (so
# the spot the agent just bombed stops being a target the moment the bomb
# is down, and the next spot / the coin becomes the target for the escape
# itself), and tiles inside an active blast are not candidates at all.
COIN_TARGET_VALUE = 3.0
CRATE_TARGET_VALUE = 1.5
TARGET_DISTANCE_WEIGHT = 0.5
BOMB_CYCLE_COST = 1.0
# feature 11's value when there is no target or no path (finite, so sklearn
# accepts it and a tree can put "unreachable" on one deterministic side)
TARGET_UNREACHABLE = float(s.COLS * s.ROWS)

# Whether the target-aware escape tie-break also runs during training (it
# always runs at evaluation). The pre-routing agent kept it evaluation-only
# (see state_features' docstring for the linear_agent history behind that
# split). With _select_target the "next target" is known at drop time, so
# the escape route is part of the routing the model must learn, not an
# afterthought -- and the screening runs bore that out: identical code
# with this False averaged 24.9 coins/round (12/50 full clears, bimodal:
# rounds that cleared fast vs rounds stuck WAITing), with it True 50.0 /
# 47.0 / 50.0 over three independent 1,500-round trainings (49, 47, 48
# full clears of 50). Escaping toward the next target is what turns the
# escape steps from pure overhead into part of the route.
TARGET_AWARE_ESCAPE_IN_TRAINING = True



# how many of the agent's own steps remain to escape after dropping a bomb
# (BOMB_TIMER counts down once per subsequent step, and the agent must be
# clear of the blast by the step it observes timer==0)
ESCAPE_BUDGET = s.BOMB_TIMER


def state_features(game_state, use_target_aware_escape=True, opponent_has_bombed=None):
    """
    Precompute the parts of the feature vector that don't depend on which
    action is being evaluated: BFS distance maps, danger/blast info, and
    the current escape route (if any). Returns None for a terminal state.

    use_target_aware_escape selects which tie-break _bfs_first_step_out_of_blast
    uses among multiple equally-short escape routes (see its docstring).
    Ported from linear_agent, including the same train/eval split: callers
    should pass False while training and True while evaluating. linear_agent
    found that retraining from scratch under the target-aware tie-break
    reliably pushed its escape-safety weights toward a different, unsafe
    optimum -- invisible at short training lengths, fully exposed at 50,000
    rounds (88% self-kill). tree_agent's escape safety is validated at 0%
    self-kill under the ORIGINAL (first-found) tie-break as of this session;
    applying the smarter routing only at evaluation time, on top of those
    already-validated weights/model, avoids reintroducing that exact risk
    rather than assuming FQI's batch-refit training is immune to it.

    opponent_has_bombed: optional {opponent_name: bool}, forwarded unchanged
    from callbacks.py's self.opponent_has_bombed (see its
    _update_opponent_bomb_history) -- read by _escape_exists_after_bomb to
    decide, per opponent, how wide a retaliation radius to simulate around
    our own candidate bomb-drop position. Defaults to "nobody has bombed
    yet" (an empty dict) if not given, which is the correct behavior at the
    very first step of a round before any history exists.
    """
    if game_state is None:
        return None

    field = game_state["field"]  # 1 for crates, -1 for stone, 0 if free
    coins = game_state["coins"]
    position = game_state["self"][3]

    danger_now, bomb_blasts, bomb_timers, danger_now_budget = _compute_danger(game_state)

    # One navigation target per step (see the COIN_TARGET_VALUE comment):
    # exactly one of coin_distance_map / crate_distance_map is set, and it
    # is a single-source BFS to that target tile, so everything downstream
    # (moves_to_coin / moves_to_crate, the potentials, the escape tie-break)
    # keeps its original gating on "which map is not None".
    target_kind, target_tile, yield_map = _select_target(
        field, position, coins, bomb_blasts, danger_now
    )
    coin_distance_map = None
    crate_distance_map = None
    if target_kind == "coin":
        coin_distance_map = _bfs_distance_map(field, [target_tile])
    elif target_kind == "spot":
        crate_distance_map = _bfs_distance_map(field, [target_tile])

    occupied = set()
    for bomb_position, _ in game_state["bombs"]:
        occupied.add(bomb_position)
    for other in game_state["others"]:
        occupied.add(other[3])

    # Task 3: multi-source BFS to the nearest opponent, mirroring
    # coin_distance_map exactly -- always computed when opponents exist
    # (opponents, unlike coins, are never hidden under a crate), independent
    # of whether a coin/crate is also being pursued this step.
    opponent_positions = [other[3] for other in game_state["others"]]
    opponent_distance_map = (
        _bfs_distance_map(field, opponent_positions) if opponent_positions else None
    )
    # Task 4: (name, position, bomb_available) per opponent -- used by
    # _escape_exists_after_bomb to account for a possible retaliatory bomb
    # from an adjacent, bomb-capable opponent (see its docstring), and by
    # callbacks.py's _adjacent_opponent_can_bomb veto. name is carried
    # through (not just position/bomb_available) so that veto can scope its
    # radius per-opponent, based on whether THIS SPECIFIC opponent has ever
    # actually been observed dropping a bomb -- see callbacks.py's
    # _update_opponent_bomb_history.
    opponents_with_bomb = [(other[0], other[3], other[2]) for other in game_state["others"]]

    # the map guiding navigation this step (see _select_target) -- used to
    # break ties among equally-short escape routes
    target_distance_map = None
    if use_target_aware_escape or TARGET_AWARE_ESCAPE_IN_TRAINING:
        target_distance_map = coin_distance_map if coin_distance_map is not None else crate_distance_map
    escape_direction, escape_distance = _compute_escape_direction(
        field, position, danger_now, bomb_blasts, bomb_timers, danger_now_budget,
        target_distance_map, opponent_positions,
    )

    # state-level gate, not itself a weighted feature: is the agent's
    # CURRENT tile inside an active threat's blast region right now?
    in_escape_window = position in danger_now or any(
        position in blast for blast in bomb_blasts.values()
    )

    return {
        "field": field,
        "position": position,
        "bomb_available": game_state["self"][2],
        "occupied": occupied,
        "coin_distance_map": coin_distance_map,
        "crate_distance_map": crate_distance_map,
        "target_kind": target_kind,
        "target_tile": target_tile,
        "yield_map": yield_map,
        "bomb_cooldown": _bomb_cooldown(game_state),
        "opponent_distance_map": opponent_distance_map,
        "opponent_positions": opponent_positions,
        "opponents_with_bomb": opponents_with_bomb,
        "opponent_has_bombed": opponent_has_bombed if opponent_has_bombed is not None else {},
        "danger_now": danger_now,
        "bomb_blasts": bomb_blasts,
        "escape_direction": escape_direction,
        "escape_distance": escape_distance,
        "in_escape_window": in_escape_window,
    }


def state_action_features(state, action):
    """
    Build phi(s, action): the N_FEATURES-dimensional feature vector for one
    candidate action, given the precomputed `state` from state_features().
    """
    if state is None:
        return np.zeros(N_FEATURES)

    position = state["position"]
    new_position = _simulate_move(position, action)
    in_escape_window = state["in_escape_window"]

    # BOMB is a structural exception: valid requires not just bomb_available
    # (see _is_valid) but a VERIFIED escape route after the drop. Diagnosed
    # empirically, not assumed: an initial port that left BOMB's validity as
    # bomb_available-only (mirroring linear_agent exactly, where
    # escape_exists_if_bomb is purely informational and the exploration
    # filter -- not the feature -- restricts real bomb-drops) produced a
    # stable ~12-15% doomed-bomb rate from round 500 onward that never
    # decayed as epsilon fell toward its floor, meaning the GREEDY policy
    # itself, not just exploration, was choosing bombs with no escape --
    # confirmed fatal 100% of the time (0% survival on doomed bombs vs.
    # 98.1% on safe ones, so the escape *detection* was never the problem).
    # A tree ensemble apparently does not reliably learn "avoid this" from a
    # ~11%-of-samples minority class the way hoped; unlike linear_agent's
    # collinearity problem this isn't fixed by giving escape_exists_if_bomb
    # more variance to learn from (ALLOW_DOOMED_BOMB_PROB already does that)
    # -- it's fixed by removing the choice structurally, the same pattern
    # that worked repeatedly elsewhere in this project (Task 1's "BOMB
    # always invalid", linear_agent's _adjacent_opponent_can_bomb).
    if action == "BOMB":
        valid = state["bomb_available"] and _escape_exists_after_bomb(state, position)
    else:
        valid = _is_valid(state, action)

    # gated to 0 whenever in_escape_window holds -- structurally removes the
    # coin-vs-danger temptation from the Q-value computation during an
    # active escape, rather than relying on a learned value to outweigh it
    moves_to_coin = 0.0
    if valid and not in_escape_window and state["coin_distance_map"] is not None:
        current_distance = state["coin_distance_map"][position]
        new_distance = state["coin_distance_map"][new_position]
        if new_distance < current_distance:
            moves_to_coin = 1.0

    # moves_to_crate: mirrors moves_to_coin, sourced from crate_distance_map
    # (the selected bomb spot, see _select_target), gated on the OPPOSITE
    # condition (coin_distance_map IS None) -- mutually exclusive with
    # moves_to_coin by construction.
    moves_to_crate = 0.0
    if valid and not in_escape_window and state["coin_distance_map"] is None and state["crate_distance_map"] is not None:
        current_distance = state["crate_distance_map"][position]
        new_distance = state["crate_distance_map"][new_position]
        if new_distance < current_distance:
            moves_to_crate = 1.0

    # an invalid move leaves the agent in place (the engine no-ops it), so
    # the tile that actually matters for danger is `position`, not the
    # unchecked `new_position`
    effective_position = new_position if valid else position

    moves_into_avoidable_danger = (
        0.0 if in_escape_window else _moves_into_danger(state, effective_position)
    )

    # valid already required escape_exists_after_bomb for BOMB above, so
    # this is just that same fact restated as a feature value, not a second
    # BFS -- kept as its own column (rather than folded into `valid`) so the
    # training log / report can still show its trajectory directly.
    escape_exists_if_bomb = 1.0 if (action == "BOMB" and valid) else 0.0
    crate_count_if_bomb = 0.0
    bomb_hits_opponent_if_bomb = 0.0
    if action == "BOMB" and valid:
        crate_count_if_bomb = float(_crate_count_if_bomb(state["field"], position))
        # MVP proxy, ported from linear_agent unchanged: does the blast
        # cover an opponent's CURRENT tile. Does not account for the
        # opponent moving before the bomb detonates (ESCAPE_BUDGET steps
        # later) -- linear_agent tried a stricter, retaliation-aware version
        # and reverted it (see report3_body.typ), so this simpler proxy is
        # ported as the validated starting point, not the untested stricter
        # one.
        blast = set(_simulate_blast_coords(position, state["field"]))
        bomb_hits_opponent_if_bomb = float(
            any(opp in blast for opp in state["opponent_positions"])
        )

    # moves_to_opponent: mirrors moves_to_coin, but deliberately NOT gated
    # to be mutually exclusive with moves_to_coin/moves_to_crate, and NOT
    # zeroed during in_escape_window -- see the N_FEATURES comment.
    moves_to_opponent = 0.0
    if valid and state["opponent_distance_map"] is not None:
        current_distance = state["opponent_distance_map"][position]
        new_distance = state["opponent_distance_map"][new_position]
        if new_distance < current_distance:
            moves_to_opponent = 1.0

    # escape_correct_move: only meaningful when in_escape_window holds --
    # does this action match the first step of the verified shortest path to
    # safety. BOMB is never valid while in_escape_window is True
    # (bomb_available only refills once the window has already closed).
    if in_escape_window and action in DIRECTIONS:
        escape_correct_move = 1.0 if state["escape_direction"] == action else 0.0
    else:
        escape_correct_move = 0.0

    # --- routing-efficiency block (11-15, see the N_FEATURES layout comment) ---
    target_map = state["coin_distance_map"]
    if target_map is None:
        target_map = state["crate_distance_map"]
    target_distance_after = _capped_distance(target_map, effective_position, TARGET_UNREACHABLE)
    target_distance_delta = target_distance_after - _capped_distance(target_map, position, TARGET_UNREACHABLE)
    yield_at_landing_tile = float(state["yield_map"].get(effective_position, 0))

    return np.array(
        [
            moves_to_coin,
            float(valid),
            1.0 if action == "WAIT" else 0.0,
            1.0 if action == "BOMB" else 0.0,
            moves_into_avoidable_danger,
            escape_exists_if_bomb,
            crate_count_if_bomb,
            escape_correct_move,
            moves_to_crate,
            moves_to_opponent,
            bomb_hits_opponent_if_bomb,
            target_distance_after,
            target_distance_delta,
            yield_at_landing_tile,
            state["bomb_cooldown"],
            1.0 if state["target_kind"] == "coin" else 0.0,
        ]
    )



def _select_target(field, position, coins, bomb_blasts, danger_now):
    """
    Pick this step's navigation target among every visible coin and every
    tile a bomb would clear at least one crate from (see the
    COIN_TARGET_VALUE comment for the scoring and the trace that motivated
    it). Returns (kind, tile, yield_map) with kind in {"coin", "spot", None}
    and yield_map = {tile: crates a bomb there would destroy} over the free
    tiles reachable from `position` (pending crates excluded), which
    state_action_features reads for yield_at_landing_tile.

    Candidates are restricted to tiles reachable from `position` over the
    static field (crates/walls only -- a bomb blocks for at most
    BOMB_TIMER steps and the escape logic already handles blasts) and
    outside every active blast / explosion: a coin or spot inside a blast
    becomes a candidate again the moment the blast has cleared. A bomb
    spot must also be one the agent could actually escape from
    (_static_escape_exists): the classic loot-crate start pocket -- three
    free tiles boxed in by crates -- makes the corner tile a 2-crate spot
    whose blast covers the whole pocket, so BOMB there is structurally
    invalid (state_action_features' escape gate) and a selector that
    still proposed it left the model with a target it could neither bomb
    nor leave without a movement penalty: 2-6% of evaluation rounds spent
    all 400 steps WAITing on the start tile. The pocket's other two tiles
    are 1-crate spots with an escape and take over.

    Ties are broken by distance, then by tile order, so the choice is a
    deterministic function of the game state (state_features is stateless
    on purpose: train.py rebuilds states from raw game_states).
    """
    threatened = set(danger_now)
    for blast in bomb_blasts.values():
        threatened.update(blast)
    pending = {tile for tile in threatened if field[tile] == 1}

    reach = _bfs_distance_map(field, [position])
    yield_map = {}
    width, height = field.shape
    for x in range(width):
        for y in range(height):
            tile = (x, y)
            if field[tile] != 0 or not np.isfinite(reach[tile]):
                continue
            crates = sum(
                1 for blast_tile in _simulate_blast_coords(tile, field)
                if field[blast_tile] == 1 and blast_tile not in pending
            )
            if crates > 0 and _static_escape_exists(field, tile):
                yield_map[tile] = crates

    best = None  # (score, -distance, kind, tile)
    for coin in coins:
        coin = (int(coin[0]), int(coin[1]))
        if coin in threatened or not np.isfinite(reach[coin]):
            continue
        distance = float(reach[coin])
        score = COIN_TARGET_VALUE - TARGET_DISTANCE_WEIGHT * distance
        candidate = (score, -distance, "coin", coin)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    for tile in sorted(yield_map):
        if tile in threatened:
            continue
        distance = float(reach[tile])
        score = yield_map[tile] * CRATE_TARGET_VALUE - TARGET_DISTANCE_WEIGHT * distance - BOMB_CYCLE_COST
        candidate = (score, -distance, "spot", tile)
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    if best is None:
        return None, None, yield_map
    return best[2], best[3], yield_map


def _static_escape_exists(field, tile):
    """
    Could a bomb dropped at `tile` be escaped from, considering only the
    static field (walls and crates)? A BFS of at most ESCAPE_BUDGET moves
    over free tiles looking for one outside the bomb's own blast -- the
    same search as _escape_exists_after_bomb without the dynamic parts
    (current blasts, opponents), so a candidate spot is never rejected for
    a threat that will be gone by the time the agent gets there, but a
    structurally inescapable pocket is always rejected.
    """
    blast = set(_simulate_blast_coords(tile, field))
    visited = {tile: 0}
    frontier = deque([tile])
    while frontier:
        current = frontier.popleft()
        depth = visited[current]
        if depth >= ESCAPE_BUDGET:
            continue
        for dx, dy in DIRECTIONS.values():
            neighbor = (current[0] + dx, current[1] + dy)
            if neighbor in visited or field[neighbor] != 0:
                continue
            visited[neighbor] = depth + 1
            if neighbor not in blast:
                return True
            frontier.append(neighbor)
    return False


def _bomb_cooldown(game_state):

    """
    Steps until this agent observes bomb_available == True again (feature
    14). Engine timing (environment.do_step: agents act, then bombs tick,
    then explosions tick): a bomb dropped at step k is observed at timers
    3, 2, 1, 0 on steps k+1..k+4, detonates during step k+4, its explosion
    is observed on k+5 (explosion_map > 0) and k+6 (no longer dangerous),
    and bomb_available is True again on k+7. So: observed own timer t ->
    t + 3; explosion still marked dangerous -> 2; otherwise -> 1.

    game_state does not say whose bomb is whose, so with opponents around
    this is a lower bound (the soonest-detonating bomb is assumed to be
    ours); exact in single-agent play.
    """
    if game_state["self"][2]:
        return 0.0
    if game_state["bombs"]:
        return float(min(timer for _pos, timer in game_state["bombs"]) + 3)
    if np.any(game_state["explosion_map"] > 0):
        return 2.0
    return 1.0


def _capped_distance(distance_map, tile, cap):
    """distance_map[tile] with None-map / unreachable (inf) mapped to `cap`."""
    if distance_map is None:
        return cap
    distance = distance_map[tile]
    return float(distance) if np.isfinite(distance) else cap


def _bfs_distance_map(field, sources):
    """
    Multi-source BFS distance map (distance to the nearest source) over the
    free tiles of `field`. Unreachable tiles are left at np.inf.
    """
    distance_map = np.full(field.shape, np.inf)
    q = deque()

    for src in sources:
        if field[src] == 0:
            distance_map[src] = 0
            q.append(src)

    while q:
        x, y = q.popleft()
        for dx, dy in DIRECTIONS.values():
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < field.shape[0]
                and 0 <= ny < field.shape[1]
                and field[nx, ny] == 0
                and distance_map[nx, ny] == np.inf
            ):
                distance_map[nx, ny] = distance_map[x, y] + 1
                q.append((nx, ny))
    return distance_map


def _is_valid(state, action):
    """True if `action` is legal from the current state."""
    field = state["field"]
    position = state["position"]
    occupied = state["occupied"]

    if action == "WAIT":
        return True
    if action == "BOMB":
        return state["bomb_available"]
    if action in DIRECTIONS:
        nx, ny = _simulate_move(position, action)
        if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
            return False
        if field[nx, ny] != 0:
            return False
        if (nx, ny) in occupied:
            return False
        return True
    return False


def _simulate_move(position, action):
    """Return the position after `action` (unchanged for WAIT/BOMB)."""
    if action in DIRECTIONS:
        dx, dy = DIRECTIONS[action]
        return (position[0] + dx, position[1] + dy)
    return position


def _simulate_blast_coords(position, field, power=s.BOMB_POWER):
    """
    Replicates items.Bomb.get_blast_coords: propagates `power` tiles in each
    cardinal direction, stopping a direction only at a stone wall (-1).
    Crates (1) do not stop the blast, they are merely destroyed by it.
    """
    x, y = position
    width, height = field.shape
    coords = [(x, y)]
    for dx, dy in DIRECTIONS.values():
        for i in range(1, power + 1):
            nx, ny = x + dx * i, y + dy * i
            if not (0 <= nx < width and 0 <= ny < height):
                break
            if field[nx, ny] == -1:
                break
            coords.append((nx, ny))
    return coords


def _compute_danger(game_state):
    """
    State-level precomputation (done once, reused for every action):
      - danger_now: tiles that are lethal *this step* if occupied.
      - bomb_blasts: for every bomb still ticking (timer > 0), the tiles its
        blast will eventually cover.
      - bomb_timers: the same ticking bombs' observed countdown values.
    """
    field = game_state["field"]
    explosion_map = game_state["explosion_map"]
    danger_now = {
        (x, y) for x, y in zip(*np.nonzero(explosion_map > 0))
    }
    danger_now_budget = {(x, y): int(explosion_map[x, y]) for (x, y) in danger_now}

    bomb_blasts = {}
    bomb_timers = {}
    for bomb_position, timer in game_state["bombs"]:
        blast = _simulate_blast_coords(bomb_position, field)
        if timer == 0:
            danger_now.update(blast)
            for tile in blast:
                danger_now_budget[tile] = 1
        else:
            bomb_blasts[bomb_position] = blast
            bomb_timers[bomb_position] = timer

    return danger_now, bomb_blasts, bomb_timers, danger_now_budget


def _moves_into_danger(state, effective_position):
    """
    True if the tile the agent actually ends up on this step is inside an
    already-dangerous explosion, or inside the future blast path of a bomb
    that hasn't exploded yet.
    """
    if effective_position in state["danger_now"]:
        return 1.0
    for blast in state["bomb_blasts"].values():
        if effective_position in blast:
            return 1.0
    return 0.0


# Manhattan-distance retaliation radius _escape_exists_after_bomb simulates
# around a candidate bomb-drop position, scoped by whether that specific
# opponent has been observed bombing this round (see the function's
# docstring). Mirrors callbacks.py's _RADIUS_IF_NEVER_BOMBED /
# _RADIUS_IF_HAS_BOMBED at the same values -- kept as separate constants
# since they govern conceptually different mechanisms (this one is
# ground-truth information the bomb-drop decision itself is gated on; that
# one is a hard veto on the BOMB action at selection time), even though
# both were validated at the same radii.
_RETALIATION_RADIUS_IF_NEVER_BOMBED = 1
_RETALIATION_RADIUS_IF_HAS_BOMBED = 2


def _escape_exists_after_bomb(state, position):
    """
    Simulates dropping a new bomb at `position` and runs a breadth-first
    search, bounded to ESCAPE_BUDGET steps, over free tiles to check whether
    at least one tile outside the new bomb's blast is reachable in time.
    Tiles that are already dangerous are excluded from the search, since
    walking through them would be fatal on the way out -- as are tiles
    currently occupied by an opponent (Task 3): an opponent's exact position
    can change before the agent gets there, but treating it as free right
    now would let this search "find" an escape through a tile that's
    actually blocked at this exact moment.

    Task 4: also accounts for a possible RETALIATORY bomb from any
    bomb-capable opponent within retaliation range of `position`. Ported
    from linear_agent, where trace analysis vs. rule_based_agent showed it
    reliably drops a bomb back at its own current tile whenever we bomb it
    while adjacent (its own rule-based logic checks exactly this condition)
    -- two simultaneous blasts from adjacent tiles routinely leave no escape
    within the fixed budget, but without this, this check only ever
    accounted for our own blast, so it looked "safe" right up until the
    opponent's own bomb closed the last exit at the same instant.
    linear_agent's own trace found this feature alone (without also pairing
    it with callbacks.py's _adjacent_opponent_can_bomb veto) didn't change
    self-kill behavior at all for a LINEAR model.

    The retaliation range itself is scoped per opponent, the same way
    callbacks.py's veto radius is: _RETALIATION_RADIUS_IF_HAS_BOMBED for an
    opponent already observed bombing this round (state["opponent_has_bombed"]),
    else _RETALIATION_RADIUS_IF_NEVER_BOMBED. Motivated directly by
    death-trace evidence (report_tree_task4_body.typ): even after the
    action-selection veto was widened, the dominant remaining self-kill
    pattern was an opponent who was close-but-not-adjacent at the moment we
    bombed, then closed the distance and bombed during our escape -- a
    threat this check, fixed at radius 1, could never see coming. Scoping
    by bombing history (rather than widening unconditionally for every
    opponent) is what keeps this a no-op against peaceful_agent, which
    never bombs and so never gets flagged.
    """
    field = state["field"]
    new_blast = set(_simulate_blast_coords(position, field))
    forbidden = set(state["danger_now"]) | set(state["opponent_positions"])
    for blast in state["bomb_blasts"].values():
        forbidden.update(blast)

    opponent_has_bombed = state["opponent_has_bombed"]
    retaliation_blast = set()
    for name, opp_pos, opp_bomb_available in state["opponents_with_bomb"]:
        if not opp_bomb_available:
            continue
        radius = (
            _RETALIATION_RADIUS_IF_HAS_BOMBED
            if opponent_has_bombed.get(name)
            else _RETALIATION_RADIUS_IF_NEVER_BOMBED
        )
        if abs(opp_pos[0] - position[0]) + abs(opp_pos[1] - position[1]) <= radius:
            retaliation_blast.update(_simulate_blast_coords(opp_pos, field))
    combined_blast = new_blast | retaliation_blast

    visited = {position: 0}
    frontier = deque([position])
    while frontier:
        current = frontier.popleft()
        depth = visited[current]
        if depth >= ESCAPE_BUDGET:
            continue
        for dx, dy in DIRECTIONS.values():
            neighbor = (current[0] + dx, current[1] + dy)
            if neighbor in visited:
                continue
            if field[neighbor] != 0 or neighbor in forbidden:
                continue
            visited[neighbor] = depth + 1
            if neighbor not in combined_blast:
                return True
            frontier.append(neighbor)
    return False


def _bfs_first_step_out_of_blast(field, start, blast, forbidden, budget, target_distance_map=None):
    """
    BFS from `start`, avoiding `forbidden` tiles, over free tiles, bounded to
    `budget` moves. Returns (DIRECTION KEY, distance) for the first move
    along the shortest path to a tile outside `blast` and that tile's
    distance from `start`, or (None, None) if no such path exists within
    budget.

    target_distance_map, if given, breaks ties among multiple equally-short
    escape routes by preferring whichever safe tile is closest to the
    current navigation target (coin_distance_map if a coin is visible, else
    crate_distance_map), rather than an arbitrary tile picked purely by
    DIRECTIONS' fixed iteration order. Ported from linear_agent, where this
    meaningfully shortened the average time to fully clear a board -- see
    state_features' docstring for why it's only applied at evaluation time.
    """
    visited = {start: None}  # tile -> first action taken from `start` to reach it
    depths = {start: 0}
    frontier = deque([start])
    safe_candidates = []  # (first_action, tile), all at the minimal escape depth
    found_depth = None

    while frontier and (found_depth is None or depths[frontier[0]] < found_depth):
        current = frontier.popleft()
        depth = depths[current]
        if depth >= budget:
            continue
        for action_name, (dx, dy) in DIRECTIONS.items():
            neighbor = (current[0] + dx, current[1] + dy)
            if neighbor in visited:
                continue
            if field[neighbor] != 0 or neighbor in forbidden:
                continue
            first_action = visited[current] if visited[current] is not None else action_name
            visited[neighbor] = first_action
            depths[neighbor] = depth + 1
            if neighbor not in blast:
                if found_depth is None:
                    found_depth = depth + 1
                safe_candidates.append((first_action, neighbor))
                continue
            frontier.append(neighbor)

    if not safe_candidates:
        return None, None
    if target_distance_map is None or len(safe_candidates) == 1:
        first_action, _ = safe_candidates[0]
        return first_action, found_depth

    def _target_distance(tile):
        d = target_distance_map[tile]
        return d if np.isfinite(d) else float("inf")

    best_action, _ = min(safe_candidates, key=lambda pair: _target_distance(pair[1]))
    return best_action, found_depth


def _compute_escape_direction(
    field, position, danger_now, bomb_blasts, bomb_timers, danger_now_budget,
    target_distance_map=None, opponent_positions=(),
):
    """
    If the agent is currently threatened, find the shortest path to safety
    and return (direction of its first step, distance to safety) -- or
    (None, None) if unthreatened, or no escape exists within budget. Two
    distinct threat sources are checked, covering the full danger window:
    an existing ticking bomb (pre-detonation phase), and the danger_now
    region itself (post-detonation, still-lingering explosion phase). Task 2
    has at most one threat active at a time (bomb_available only resets
    once a bomb's danger fully clears), so at most one branch fires.

    opponent_positions (Task 3): treated as forbidden in both branches, for
    the same reason _escape_exists_after_bomb does -- an opponent occupies
    its current tile right now, so a route planned through it isn't
    actually executable this step.
    """
    for bomb_pos, timer in bomb_timers.items():
        blast = bomb_blasts[bomb_pos]
        if position not in blast:
            continue
        remaining_budget = timer + 1
        forbidden = set(danger_now) | set(opponent_positions)
        for other_pos, other_blast in bomb_blasts.items():
            if other_pos != bomb_pos:
                forbidden.update(other_blast)
        first_step, distance = _bfs_first_step_out_of_blast(
            field, position, set(blast), forbidden, remaining_budget, target_distance_map
        )
        if first_step is not None:
            return first_step, distance

    if position in danger_now:
        remaining_budget = danger_now_budget[position]
        forbidden = set(opponent_positions)
        for blast in bomb_blasts.values():
            forbidden.update(blast)
        first_step, distance = _bfs_first_step_out_of_blast(
            field, position, set(danger_now), forbidden, remaining_budget, target_distance_map
        )
        if first_step is not None:
            return first_step, distance

    return None, None


def _crate_count_if_bomb(field, position):
    """How many crates a bomb dropped at `position` would destroy."""
    blast = _simulate_blast_coords(position, field)
    return sum(1 for x, y in blast if field[x, y] == 1)


def coin_potential(state):
    """
    Potential-based shaping term: negative BFS distance to the selected coin
    target (Ng, Harada & Russell 1999 -- provably policy-invariant; the
    target is a deterministic function of the state, see _select_target).
    """
    if state is None or state["coin_distance_map"] is None:
        return 0.0

    distance = state["coin_distance_map"][state["position"]]

    if np.isfinite(distance):
        return -distance
    else:
        return 0.0


# Scales opponent_potential's raw BFS-distance contribution relative to
# coin_potential/crate_potential's implicit scale of 1.0. A positive scalar
# multiple of a valid potential function is itself still a valid
# (policy-invariant) potential function.
#
# Diagnosed as necessary, not assumed: a first Task 3 training run at the
# unscaled default (1.0, mirroring linear_agent's own unchanged default)
# produced a severe regression -- a 3-round direct trace found the greedy
# policy dropping ZERO bombs and collecting ZERO coins across 1,200 total
# steps, with Q(BOMB) losing the argmax to movement from the very first
# step of a round (Q(BOMB)=45.67 vs Q(LEFT)=51.70), even though BOMB was
# structurally valid on 100% of those steps. Root cause: unlike
# coin_potential/crate_potential (mutually exclusive with each other, and
# comparatively sparse), opponent_potential is unconditionally active on
# every single step -- it doubles the "always-on" shaping density relative
# to Task 2's baseline, while CRATE_HIT_BONUS/EXTRA_CRATE_BONUS only land on
# the comparatively rare BOMB transitions themselves. In Fitted
# Q-Iteration's batch regression over the replay buffer, that density
# imbalance let the dense opponent-distance signal dominate the learned
# value function, crowding out the sparser bombing incentive almost
# entirely -- a different mechanism from, but the same underlying shape as,
# this project's other reward-density regressions (Task 2's WAIT-collapse,
# the EXTRA_CRATE_BONUS overcorrection). linear_agent has an identical
# knob, tried at 0.3 for a different reason (protecting
# escape_correct_move, not is_bomb) and reverted since it didn't fix that
# specific problem -- ported here as a genuinely new fix for a
# tree_agent-specific failure mode, not a straight port of linear_agent's
# own (unsuccessful, for its own purposes) attempt at the same lever.
OPPONENT_POTENTIAL_SCALE = 0.2


def opponent_potential(state):
    """
    Potential-based shaping term (Task 3): negative BFS distance to the
    nearest opponent, scaled by OPPONENT_POTENTIAL_SCALE (see its comment
    for why the scale-down was necessary). Unlike coin_potential/
    crate_potential, always active when at least one opponent exists --
    hunting is layered on top of coin/crate seeking, not gated to be
    exclusive with it (see the N_FEATURES comment on moves_to_opponent).
    """
    if state is None or state["opponent_distance_map"] is None:
        return 0.0

    distance = state["opponent_distance_map"][state["position"]]

    if np.isfinite(distance):
        return -distance * OPPONENT_POTENTIAL_SCALE
    else:
        return 0.0


def crate_potential(state):
    """
    Potential-based shaping term for crate-seeking: negative BFS distance to
    the selected bomb-spot target. Mirrors coin_potential, gated to be
    active only when coin_potential is NOT (the selected target is a bomb
    spot, not a coin) -- the same condition moves_to_crate is gated on.
    """
    if state is None or state["coin_distance_map"] is not None or state["crate_distance_map"] is None:
        return 0.0

    distance = state["crate_distance_map"][state["position"]]

    if np.isfinite(distance):
        return -distance
    else:
        return 0.0


def escape_potential(state):
    """
    Potential-based shaping term while escaping: negative BFS distance to
    the nearest tile outside the current threat's blast, active only while
    in_escape_window holds. Reuses state["escape_distance"] (the same BFS
    that computes escape_direction) rather than running a second search.
    """
    if state is None or not state["in_escape_window"]:
        return 0.0

    distance = state["escape_distance"]

    if distance is None:
        return 0.0
    else:
        return -distance


def q_values(model, state):
    """
    Q(s, a) for every action in ACTIONS. Returns all zeros if state is
    terminal or if model is None (before the first Fitted-Q-Iteration
    refit).
    """
    values = np.zeros(len(ACTIONS))

    if state is None or model is None:
        return values

    phis = np.stack([state_action_features(state, action) for action in ACTIONS])
    values[:] = model.predict(phis)
    return values
