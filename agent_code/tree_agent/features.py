"""
Features and Q-function of the tree_agent.

Q(s, a) = model.predict(phi(s, a)), one regressor for all six actions
(phi is evaluated once per candidate action). The rest of the file is plain
game-state computation: BFS maps, blast simulation, escape routing, target
selection and the potentials used for reward shaping.
"""

from collections import deque

import numpy as np

import settings as s

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]

# image coordinates, top left is (0,0)
DIRECTIONS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

# phi(s, a) layout. The shipped model was fit on exactly this order.
F_MOVES_TO_COIN, F_VALID, F_IS_WAIT, F_IS_BOMB, F_INTO_DANGER = 0, 1, 2, 3, 4
F_ESCAPE_IF_BOMB, F_CRATES_IF_BOMB, F_ESCAPE_MOVE, F_MOVES_TO_CRATE = 5, 6, 7, 8
F_MOVES_TO_OPPONENT, F_HITS_OPPONENT = 9, 10
F_TARGET_DIST, F_TARGET_DELTA, F_YIELD_HERE, F_COOLDOWN, F_TARGET_IS_COIN = 11, 12, 13, 14, 15
#  0 moves_to_coin / 8 moves_to_crate: step reduces the BFS distance to the
#    navigation target (a coin or a bomb spot, see _select_target); which of
#    the two is set depends on the target kind, both are 0 during an escape
#  4 moves_into_avoidable_danger / 7 escape_correct_move: the first is used
#    outside an escape window, the second inside it (the other is forced 0)
#  6 crate_count_if_bomb: number of crates the bomb would destroy
#  9 moves_to_opponent: like 0 but towards the nearest opponent, never gated
# 11 BFS distance from the landing tile to the target (TARGET_UNREACHABLE if none)
# 12 change of that distance: -1 toward, +1 away, 0 for WAIT/BOMB/invalid
# 13 crates a bomb on the landing tile would destroy
# 14 steps until bomb_available is back (0 if it is now)
# 15 target is a coin (1) or a bomb spot / nothing (0)
N_FEATURES = 16
GAMMA = 0.95

# Navigation target: every visible coin and every tile a bomb would clear at
# least one crate from is scored, the best one is THE target for this step:
#   coin:  COIN_TARGET_VALUE - TARGET_DISTANCE_WEIGHT * distance
#   spot:  yield * CRATE_TARGET_VALUE - TARGET_DISTANCE_WEIGHT * distance - BOMB_CYCLE_COST
# So a 3-crate spot under the agent beats a coin two steps away, a coin next
# door beats any 1-crate spot.
COIN_TARGET_VALUE = 3.0
# With opponents around coins are first come first served, so they outrank
# crates by a wide margin (a coin beats a 3-crate spot underfoot out to ~11 tiles).
COIN_TARGET_VALUE_VS_OPPONENTS = 9.0
CRATE_TARGET_VALUE = 1.5
TARGET_DISTANCE_WEIGHT = 0.5
BOMB_CYCLE_COST = 1.0
# value of feature 11 without a target or path; finite so the trees can split on it
TARGET_UNREACHABLE = float(s.COLS * s.ROWS)

# How hunting interacts with the away-penalty of train.move_bonus. Either
# exempt steps toward an opponent from the penalty, or make opponents
# targets in their own right (OPPONENT_TARGET_VALUE - weight * distance).
# Both off: hunting is driven by opponent_potential and feature 9 alone.
HUNT_EXEMPT_FROM_AWAY_PENALTY = False
OPPONENTS_AS_TARGETS = False
OPPONENT_TARGET_VALUE = 6.0

# Break escape-route ties toward the navigation target during training as
# well (it always applies at evaluation). Without it training was bimodal:
# some runs cleared boards fast, others got stuck waiting.
TARGET_AWARE_ESCAPE_IN_TRAINING = True

# own steps left to get clear after dropping a bomb
ESCAPE_BUDGET = s.BOMB_TIMER


def state_features(game_state, use_target_aware_escape=True, opponent_has_bombed=None):
    """
    Action-independent part of one step: distance maps, threats, target and
    escape route. None for a terminal state. opponent_has_bombed is the
    {name: bool} record callbacks.py keeps of who has bombed this round.
    """
    if game_state is None:
        return None

    field = game_state["field"]  # 1 crate, -1 stone, 0 free
    coins = game_state["coins"]
    position = game_state["self"][3]

    lethal_now, bomb_blasts, bomb_timers, lethal_for = _threat_map(game_state)

    # one target per step; exactly one of coin_distance_map / crate_distance_map is set
    opponent_tiles = [other[3] for other in game_state["others"]]
    target_kind, target_tile, yield_map = _select_target(
        field, position, coins, bomb_blasts, lethal_now,
        opponent_positions=opponent_tiles if OPPONENTS_AS_TARGETS else (),
        opponents_present=bool(opponent_tiles),
    )

    coin_distance_map = None
    crate_distance_map = None
    target_distance_map = None
    if target_kind is not None:
        target_distance_map = _distance_map(field, [target_tile])
    if target_kind == "coin":
        coin_distance_map = target_distance_map
    elif target_kind == "spot":
        crate_distance_map = target_distance_map
    # an opponent target sets neither map; only 11-12, move_bonus and the
    # escape tie-break follow it

    occupied = set()
    for bomb_position, _ in game_state["bombs"]:
        occupied.add(bomb_position)
    for other in game_state["others"]:
        occupied.add(other[3])

    opponent_positions = [other[3] for other in game_state["others"]]
    opponent_distance_map = (
        _distance_map(field, opponent_positions) if opponent_positions else None
    )
    # (name, position, bomb_available) per opponent
    opponents = [(other[0], other[3], other[2]) for other in game_state["others"]]

    escape_target_map = None
    if use_target_aware_escape or TARGET_AWARE_ESCAPE_IN_TRAINING:
        escape_target_map = target_distance_map
    likely_blast = _likely_opponent_blast(field, opponents, opponent_has_bombed or {})
    escape_direction, escape_distance = _find_escape(
        field, position, lethal_now, bomb_blasts, bomb_timers, lethal_for,
        escape_target_map, opponent_positions, likely_blast,
    )

    in_escape_window = position in lethal_now or any(
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
        "target_distance_map": target_distance_map,
        "yield_map": yield_map,
        "bomb_cooldown": _bomb_cooldown(game_state),
        "opponent_distance_map": opponent_distance_map,
        "opponent_positions": opponent_positions,
        "opponents": opponents,
        "opponent_has_bombed": opponent_has_bombed if opponent_has_bombed is not None else {},
        "lethal_now": lethal_now,
        "bomb_blasts": bomb_blasts,
        "bomb_timers": bomb_timers,
        "escape_direction": escape_direction,
        "escape_distance": escape_distance,
        "in_escape_window": in_escape_window,
    }


def state_action_features(state, action):
    if state is None:
        return np.zeros(N_FEATURES)

    position = state["position"]
    new_position = _step(position, action)
    in_escape_window = state["in_escape_window"]

    # BOMB is only valid with a verified escape route. Leaving that to the
    # model produced a stable ~12% rate of bombs with no way out.
    if action == "BOMB":
        valid = state["bomb_available"] and _escape_exists_after_bomb(state, position)
    else:
        valid = _legal(state, action)

    # no coin/crate credit while escaping
    moves_to_coin = 0.0
    if valid and not in_escape_window and state["coin_distance_map"] is not None:
        current_distance = state["coin_distance_map"][position]
        new_distance = state["coin_distance_map"][new_position]
        if new_distance < current_distance:
            moves_to_coin = 1.0

    moves_to_crate = 0.0
    if valid and not in_escape_window and state["coin_distance_map"] is None and state["crate_distance_map"] is not None:
        current_distance = state["crate_distance_map"][position]
        new_distance = state["crate_distance_map"][new_position]
        if new_distance < current_distance:
            moves_to_crate = 1.0

    # an invalid move leaves the agent where it is
    effective_position = new_position if valid else position

    moves_into_avoidable_danger = (
        0.0 if in_escape_window else _tile_is_threatened(state, effective_position)
    )

    escape_exists_if_bomb = 1.0 if (action == "BOMB" and valid) else 0.0
    crate_count_if_bomb = 0.0
    bomb_hits_opponent_if_bomb = 0.0
    if action == "BOMB" and valid:
        crate_count_if_bomb = float(_crates_hit_from(state["field"], position))
        # does the blast cover an opponent's current tile
        blast = set(_blast_tiles(position, state["field"]))
        bomb_hits_opponent_if_bomb = float(
            any(opp in blast for opp in state["opponent_positions"])
        )

    moves_to_opponent = 0.0
    if valid and state["opponent_distance_map"] is not None:
        current_distance = state["opponent_distance_map"][position]
        new_distance = state["opponent_distance_map"][new_position]
        if new_distance < current_distance:
            moves_to_opponent = 1.0

    if in_escape_window and action in DIRECTIONS:
        escape_correct_move = 1.0 if state["escape_direction"] == action else 0.0
    else:
        escape_correct_move = 0.0

    target_map = state["target_distance_map"]
    target_distance_after = _dist_or(target_map, effective_position, TARGET_UNREACHABLE)
    target_distance_delta = target_distance_after - _dist_or(target_map, position, TARGET_UNREACHABLE)
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


def _select_target(field, position, coins, bomb_blasts, lethal_now, opponent_positions=(), opponents_present=False):
    """
    Returns (kind, tile, yield_map), kind in {"coin", "spot", "opponent", None}.
    yield_map: reachable free tile -> crates a bomb there would destroy
    (crates already inside a blast don't count). Candidates must be reachable,
    outside every blast and, for spots, escapable on the static field --
    otherwise the boxed-in start pocket offers a spot we can neither bomb nor leave.
    """
    threatened = set(lethal_now)
    for blast in bomb_blasts.values():
        threatened.update(blast)
    pending = {tile for tile in threatened if field[tile] == 1}

    reach = _distance_map(field, [position])
    yield_map = {}
    width, height = field.shape
    for x in range(width):
        for y in range(height):
            tile = (x, y)
            if field[tile] != 0 or not np.isfinite(reach[tile]):
                continue
            crates = sum(
                1 for blast_tile in _blast_tiles(tile, field)
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
        coin_value = COIN_TARGET_VALUE_VS_OPPONENTS if opponents_present else COIN_TARGET_VALUE
        score = coin_value - TARGET_DISTANCE_WEIGHT * distance
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

    for opp in opponent_positions:
        opp = (int(opp[0]), int(opp[1]))
        if opp in threatened or not np.isfinite(reach[opp]):
            continue
        distance = float(reach[opp])
        score = OPPONENT_TARGET_VALUE - TARGET_DISTANCE_WEIGHT * distance
        candidate = (score, -distance, "opponent", opp)
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    if best is None:
        return None, None, yield_map
    return best[2], best[3], yield_map


# like _escape_exists_after_bomb but on walls and crates only
def _static_escape_exists(field, tile):
    blast = set(_blast_tiles(tile, field))
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


# steps until bomb_available is True again. Engine order is act, bombs tick,
# explosions tick: a bomb dropped at step k shows timers 3,2,1,0 on k+1..k+4,
# its explosion is dangerous on k+5, harmless on k+6, bomb back on k+7.
# With opponents around this is a lower bound (the soonest bomb may not be ours).
def _bomb_cooldown(game_state):
    if game_state["self"][2]:
        return 0.0
    if game_state["bombs"]:
        return float(min(timer for _pos, timer in game_state["bombs"]) + 3)
    if np.any(game_state["explosion_map"] > 0):
        return 2.0
    return 1.0


def _dist_or(distance_map, tile, default):
    if distance_map is None:
        return default
    distance = distance_map[tile]
    return float(distance) if np.isfinite(distance) else default


# multi-source BFS over free tiles, unreachable = inf
def _distance_map(field, sources):
    distance_map = np.full(field.shape, np.inf)
    queue = deque()

    for source in sources:
        if field[source] == 0:
            distance_map[source] = 0
            queue.append(source)

    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRECTIONS.values():
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < field.shape[0]
                and 0 <= ny < field.shape[1]
                and field[nx, ny] == 0
                and distance_map[nx, ny] == np.inf
            ):
                distance_map[nx, ny] = distance_map[x, y] + 1
                queue.append((nx, ny))
    return distance_map


def _legal(state, action):
    field = state["field"]
    position = state["position"]
    occupied = state["occupied"]

    if action == "WAIT":
        return True
    if action == "BOMB":
        return state["bomb_available"]
    if action in DIRECTIONS:
        nx, ny = _step(position, action)
        if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
            return False
        if field[nx, ny] != 0:
            return False
        if (nx, ny) in occupied:
            return False
        return True
    return False


def _step(position, action):
    if action in DIRECTIONS:
        dx, dy = DIRECTIONS[action]
        return (position[0] + dx, position[1] + dy)
    return position


# same as items.Bomb.get_blast_coords: stone stops the blast, crates don't
def _blast_tiles(position, field, power=s.BOMB_POWER):
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


def _threat_map(game_state):
    """
    (lethal_now, bomb_blasts, bomb_timers, lethal_for): tiles that kill this
    step, blast and timer of every ticking bomb, and how long each lethal
    tile stays lethal.
    """
    field = game_state["field"]
    explosion_map = game_state["explosion_map"]
    lethal_now = {
        (x, y) for x, y in zip(*np.nonzero(explosion_map > 0))
    }
    lethal_for = {(x, y): int(explosion_map[x, y]) for (x, y) in lethal_now}

    bomb_blasts = {}
    bomb_timers = {}
    for bomb_position, timer in game_state["bombs"]:
        blast = _blast_tiles(bomb_position, field)
        if timer == 0:
            lethal_now.update(blast)
            for tile in blast:
                lethal_for[tile] = 1
        else:
            bomb_blasts[bomb_position] = blast
            bomb_timers[bomb_position] = timer

    return lethal_now, bomb_blasts, bomb_timers, lethal_for


def _tile_is_threatened(state, tile):
    if tile in state["lethal_now"]:
        return 1.0
    for blast in state["bomb_blasts"].values():
        if tile in blast:
            return 1.0
    return 0.0


# Manhattan radius within which a bomb-capable opponent is assumed to bomb
# back when we drop one: 1 tile normally, 2 once that opponent has been seen
# bombing this round. Same values as the veto radius in callbacks.py, kept
# separate because this gates the feature and that gates the action.
RETALIATION_RADIUS_QUIET = 1
RETALIATION_RADIUS_ACTIVE = 2

# Escape-search refinements, each behind a flag so it can be ablated:
#   ESCAPE_TIMING_AWARE: a tile in another bomb's blast may be crossed while
#     that bomb's timer allows (deadly from depth timer + 1), never stopped on
#   ESCAPE_AVOIDS_OPPONENT_REACH: tiles an opponent can step into next turn
#     are blocked; at drop time this can veto the bomb, in routing it is a
#     first pass with the plain search as fallback
#   ESCAPE_AVOIDS_PLAUSIBLE_OPPONENT_BLAST: don't end the route where an
#     opponent seen bombing this round could bomb us right now (first pass only)
ESCAPE_TIMING_AWARE = True
ESCAPE_AVOIDS_OPPONENT_REACH = True
ESCAPE_AVOIDS_PLAUSIBLE_OPPONENT_BLAST = True


# tile -> last search depth at which a route may still enter it
def _blast_deadlines(bomb_blasts, bomb_timers, exclude=None):
    deadlines = {}
    for bomb_pos, blast in bomb_blasts.items():
        if bomb_pos == exclude:
            continue
        timer = bomb_timers[bomb_pos]
        for tile in blast:
            deadlines[tile] = min(deadlines.get(tile, timer), timer)
    return deadlines


# opponents' tiles plus wherever they can step next turn
def _opponent_reach(field, opponent_positions):
    reach = set()
    for opp in opponent_positions:
        opp = (int(opp[0]), int(opp[1]))
        reach.add(opp)
        for dx, dy in DIRECTIONS.values():
            tile = (opp[0] + dx, opp[1] + dy)
            if 0 <= tile[0] < field.shape[0] and 0 <= tile[1] < field.shape[1] and field[tile] == 0:
                reach.add(tile)
    return reach


def _likely_opponent_blast(field, opponents, opponent_has_bombed):
    blast = set()
    for name, opp_pos, opp_bomb_available in opponents:
        if opp_bomb_available and opponent_has_bombed.get(name):
            blast.update(_blast_tiles(opp_pos, field))
    return blast


def _escape_exists_after_bomb(state, position):
    """
    Is there a way out within ESCAPE_BUDGET moves if we bomb `position` now?
    Also accounts for a bomb dropped back by an opponent within
    RETALIATION_RADIUS_* and for other bombs already ticking.
    """
    field = state["field"]
    new_blast = set(_blast_tiles(position, field))
    forbidden = set(state["lethal_now"]) | set(state["opponent_positions"])
    if ESCAPE_AVOIDS_OPPONENT_REACH:
        forbidden |= _opponent_reach(field, state["opponent_positions"])
    other_blast_tiles = set()
    for blast in state["bomb_blasts"].values():
        other_blast_tiles.update(blast)
    if ESCAPE_TIMING_AWARE:
        deadlines = _blast_deadlines(state["bomb_blasts"], state["bomb_timers"])
    else:
        deadlines = {}
        forbidden |= other_blast_tiles

    opponent_has_bombed = state["opponent_has_bombed"]
    retaliation_blast = set()
    for name, opp_pos, opp_bomb_available in state["opponents"]:
        if not opp_bomb_available:
            continue
        radius = (
            RETALIATION_RADIUS_ACTIVE
            if opponent_has_bombed.get(name)
            else RETALIATION_RADIUS_QUIET
        )
        if abs(opp_pos[0] - position[0]) + abs(opp_pos[1] - position[1]) <= radius:
            retaliation_blast.update(_blast_tiles(opp_pos, field))
    combined_blast = new_blast | retaliation_blast | other_blast_tiles

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
            if neighbor in deadlines and depth + 1 > deadlines[neighbor]:
                continue
            visited[neighbor] = depth + 1
            if neighbor not in combined_blast:
                return True
            frontier.append(neighbor)
    return False


def _escape_route(
    field, start, blast, forbidden, budget, target_distance_map=None,
    deadlines=None, unsafe_destinations=frozenset(),
):
    """
    BFS out of `blast`: (first direction, distance) or (None, None). Ties
    between equally short routes go to the tile closest to the target if a
    map is given. unsafe_destinations may be crossed but not ended on.
    """
    if deadlines is None:
        deadlines = {}
    visited = {start: None}  # tile -> first action on the way there
    depths = {start: 0}
    frontier = deque([start])
    safe_candidates = []  # (first_action, tile) at the minimal depth
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
            if neighbor in deadlines and depth + 1 > deadlines[neighbor]:
                continue
            first_action = visited[current] if visited[current] is not None else action_name
            visited[neighbor] = first_action
            depths[neighbor] = depth + 1
            if neighbor not in blast and neighbor not in unsafe_destinations:
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


def _find_escape(
    field, position, lethal_now, bomb_blasts, bomb_timers, lethal_for,
    target_distance_map=None, opponent_positions=(), likely_opponent_blast=frozenset(),
):
    """
    (first step, distance) to safety if we are threatened, else (None, None).
    Threats: a ticking bomb covering us (budget timer + 1) or a lingering
    explosion. A conservative pass (opponent reach / likely blast excluded)
    runs first, the plain search is the fallback, so no route is ever lost.
    """
    base_forbidden = set(lethal_now) | set(opponent_positions)
    passes = [(set(), set())]
    conservative_forbidden = _opponent_reach(field, opponent_positions) if ESCAPE_AVOIDS_OPPONENT_REACH else set()
    conservative_unsafe = set(likely_opponent_blast) if ESCAPE_AVOIDS_PLAUSIBLE_OPPONENT_BLAST else set()
    if conservative_forbidden or conservative_unsafe:
        passes.insert(0, (conservative_forbidden, conservative_unsafe))

    for bomb_pos, timer in bomb_timers.items():
        blast = bomb_blasts[bomb_pos]
        if position not in blast:
            continue
        remaining_budget = timer + 1
        other_blast_tiles = set()
        for other_pos, other_blast in bomb_blasts.items():
            if other_pos != bomb_pos:
                other_blast_tiles.update(other_blast)
        deadlines = _blast_deadlines(bomb_blasts, bomb_timers, exclude=bomb_pos) if ESCAPE_TIMING_AWARE else {}
        for extra_forbidden, extra_unsafe in passes:
            forbidden = base_forbidden | extra_forbidden
            if not ESCAPE_TIMING_AWARE:
                forbidden |= other_blast_tiles
            forbidden.discard(position)  # we are standing there
            first_step, distance = _escape_route(
                field, position, set(blast), forbidden, remaining_budget, target_distance_map,
                deadlines=deadlines, unsafe_destinations=other_blast_tiles | extra_unsafe,
            )
            if first_step is not None:
                return first_step, distance

    if position in lethal_now:
        remaining_budget = lethal_for[position]
        all_blast_tiles = set()
        for blast in bomb_blasts.values():
            all_blast_tiles.update(blast)
        deadlines = _blast_deadlines(bomb_blasts, bomb_timers) if ESCAPE_TIMING_AWARE else {}
        for extra_forbidden, extra_unsafe in passes:
            forbidden = set(opponent_positions) | extra_forbidden
            if not ESCAPE_TIMING_AWARE:
                forbidden |= all_blast_tiles
            forbidden.discard(position)
            first_step, distance = _escape_route(
                field, position, set(lethal_now), forbidden, remaining_budget, target_distance_map,
                deadlines=deadlines, unsafe_destinations=all_blast_tiles | extra_unsafe,
            )
            if first_step is not None:
                return first_step, distance

    return None, None


def _crates_hit_from(field, position):
    blast = _blast_tiles(position, field)
    return sum(1 for x, y in blast if field[x, y] == 1)


# Potentials for reward shaping (Ng, Harada & Russell 1999), each minus a BFS
# distance. coin/crate are exclusive (whichever the target is), escape only
# inside a blast, opponent always when one exists.

def coin_potential(state):
    if state is None or state["coin_distance_map"] is None:
        return 0.0

    distance = state["coin_distance_map"][state["position"]]

    if np.isfinite(distance):
        return -distance
    else:
        return 0.0


# opponent_potential is active on every step, unlike the other three; at
# scale 1.0 it swamped the bombing incentive (the greedy policy stopped
# dropping bombs altogether).
OPPONENT_POTENTIAL_SCALE = 0.2


def opponent_potential(state):
    if state is None or state["opponent_distance_map"] is None:
        return 0.0

    distance = state["opponent_distance_map"][state["position"]]

    if np.isfinite(distance):
        return -distance * OPPONENT_POTENTIAL_SCALE
    else:
        return 0.0


def crate_potential(state):
    if state is None or state["coin_distance_map"] is not None or state["crate_distance_map"] is None:
        return 0.0

    distance = state["crate_distance_map"][state["position"]]

    if np.isfinite(distance):
        return -distance
    else:
        return 0.0


def escape_potential(state):
    if state is None or not state["in_escape_window"]:
        return 0.0

    distance = state["escape_distance"]

    if distance is None:
        return 0.0
    else:
        return -distance


def q_values(model, state):
    values = np.zeros(len(ACTIONS))

    if state is None or model is None:
        return values

    phis = np.stack([state_action_features(state, action) for action in ACTIONS])
    values[:] = model.predict(phis)
    return values
