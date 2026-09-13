"""
State representation and the linear Q-function for the linear_agent.

The agent is a linear function approximator: Q(s, a) = weights . phi(s, a),
where phi(s, a) is the 9-dimensional feature vector state_action_features()
returns below. There is no neural network, replay buffer, or target
network -- self.weights (see callbacks.py) is the entire model.

Public API used by callbacks.py / train.py:
    state_features(game_state)            -- precompute the state-level parts
                                              shared by every action this step
    state_action_features(state, action)  -- the feature vector for one action
    q_values(weights, state)              -- Q(s, .) for every action
    coin_potential / crate_potential /
    escape_potential(state)               -- potential-based reward-shaping
                                              terms, summed in train.py
    N_FEATURES, GAMMA, ESCAPE_BUDGET       -- shared constants

Everything else in this file is a private helper (leading underscore).
"""

from collections import deque

import numpy as np

import settings as s

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]

# image coordinates, top left is (0,0)
DIRECTIONS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

# feature vector layout (index: name)
# 0: moves_to_coin          5: escape_exists_if_bomb
# 1: valid                  6: crate_hit_if_bomb
# 2: is_wait                7: escape_correct_move
# 3: is_bomb                8: moves_to_crate
# 4: moves_into_avoidable_danger
#
# 4 and 7 are mutually exclusive by construction (gated on in_escape_window,
# see state_features): each is forced to 0 whenever the other's context is
# active, so a single weight never has to represent both "avoid unforced
# danger" and "take the correct escape step" at once.
#
# 0 and 8 are likewise mutually exclusive by construction (gated on whether
# coin_distance_map is None, see state_features): moves_to_crate gives
# directional credit toward the nearest crate on steps where no coin is
# visible and moves_to_coin has nothing to go on -- it mirrors moves_to_coin
# exactly, just sourced from crate_distance_map instead.
N_FEATURES = 9
GAMMA = 0.95  # discount factor

# how many of the agent's own steps remain to escape after dropping a bomb
# (BOMB_TIMER counts down once per subsequent step, and the agent must be
# clear of the blast by the step it observes timer==0)
ESCAPE_BUDGET = s.BOMB_TIMER


def state_features(game_state, use_target_aware_escape=True):
    """
    Precompute the parts of the feature vector that don't depend on which
    action is being evaluated: BFS distance maps, danger/blast info, and
    the current escape route (if any). Returns None for a terminal state.

    use_target_aware_escape selects which tie-break _compute_escape_direction
    uses among multiple equally-short escape routes (see its docstring).
    Callers should pass False while training and True while evaluating: the
    saved model's weights were trained under the plain (non-target-aware)
    tie-break, and retraining from scratch under the target-aware one
    changes the training-time state distribution enough to reliably drive
    the escape-safety weights into a different, unsafe optimum (confirmed
    at 50,000 rounds: 88% self-kill, despite looking fine at 5,000 and
    20,000). The target-aware routing is still a genuine improvement at
    evaluation time on top of weights trained the original way -- it just
    isn't safe to train under yet.
    """

    if game_state is None:
        return None

    field = game_state["field"]  # 1 for crates, -1 for stone, 0 if free
    coins = game_state["coins"]  # coin coordinate list

    if coins:
        coin_distance_map = _bfs_distance_map(field, coins)
        crate_distance_map = None
    else:
        coin_distance_map = None
        # only computed when there's no coin to seek -- moves_to_crate and
        # crate_potential are both gated to be active exactly when
        # coin_potential/moves_to_coin are not, so there's no point paying
        # for this BFS on every step a coin is already providing guidance
        crate_distance_map = _bfs_distance_map(field, _crate_adjacent_tiles(field))

    occupied = set()
    for bomb_position, _ in game_state["bombs"]:
        occupied.add(bomb_position)
    for other in game_state["others"]:
        occupied.add(other[3])

    danger_now, bomb_blasts, bomb_timers, danger_now_budget = _compute_danger(game_state)
    position = game_state["self"][3]
    # whichever map is currently guiding navigation (coin takes priority,
    # matching moves_to_coin/moves_to_crate's own gating) -- used to break
    # ties among equally-short escape routes, see
    # _bfs_first_step_out_of_blast for why
    target_distance_map = None
    if use_target_aware_escape:
        target_distance_map = coin_distance_map if coin_distance_map is not None else crate_distance_map
    escape_direction, escape_distance = _compute_escape_direction(
        field, position, danger_now, bomb_blasts, bomb_timers, danger_now_budget, target_distance_map
    )

    # state-level gate, not itself a weighted feature: is the agent's
    # CURRENT tile inside an active threat's blast region right now (a
    # ticking bomb it placed, or that same bomb's still-lingering
    # explosion)? escape_direction can be None either because there's no
    # threat, or because a threat exists but no escape fits within budget;
    # in_escape_window disambiguates those two by checking the threat
    # condition directly, independent of whether a path out was found.
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
        "danger_now": danger_now,
        "bomb_blasts": bomb_blasts,
        "escape_direction": escape_direction,
        "escape_distance": escape_distance,
        "in_escape_window": in_escape_window,
    }


def state_action_features(state, action):
    """
    Build phi(s, action): the 9-dimensional feature vector for one
    candidate action, given the precomputed `state` from state_features().
    """
    if state is None:
        return np.zeros(N_FEATURES)

    valid = _is_valid(state, action)
    position = state["position"]
    new_position = _simulate_move(position, action)
    in_escape_window = state["in_escape_window"]
    moves_to_coin = 0.0

    # gated to 0 whenever in_escape_window holds, regardless of the
    # underlying BFS-distance comparison -- structurally removes the
    # coin-vs-danger temptation from the Q-value computation during an
    # active escape, rather than relying on a learned weight to outweigh it
    if valid and not in_escape_window and state["coin_distance_map"] is not None:
        current_distance = state["coin_distance_map"][position]
        new_distance = state["coin_distance_map"][new_position]

        if new_distance < current_distance:
            moves_to_coin = 1.0

    # moves_to_crate: mirrors moves_to_coin exactly, but sourced from
    # crate_distance_map and gated on the OPPOSITE condition
    # (coin_distance_map IS None) -- mutually exclusive with moves_to_coin
    # by construction, giving the model a place to attach directional
    # credit toward the nearest crate on the steps where no coin is visible.
    moves_to_crate = 0.0
    if valid and not in_escape_window and state["coin_distance_map"] is None and state["crate_distance_map"] is not None:
        current_distance = state["crate_distance_map"][position]
        new_distance = state["crate_distance_map"][new_position]

        if new_distance < current_distance:
            moves_to_crate = 1.0

    # an invalid move leaves the agent in place (the engine no-ops it), so the
    # tile that actually matters for danger is `position`, not the unchecked
    # `new_position` (which may point into a wall/crate for invalid actions)
    effective_position = new_position if valid else position

    # moves_into_avoidable_danger: only evaluated when NOT already forced to
    # be dealing with a threat this step (in_escape_window == False) -- an
    # unnecessary, avoidable entry into danger. Mutually exclusive with
    # escape_correct_move below by construction (each requires the opposite
    # value of in_escape_window), so neither weight ever sees samples from
    # the other's context.
    moves_into_avoidable_danger = (
        0.0 if in_escape_window else _moves_into_danger(state, effective_position)
    )

    escape_exists_if_bomb = 0.0
    crate_hit_if_bomb = 0.0
    if action == "BOMB" and valid:
        escape_exists_if_bomb = float(_escape_exists_after_bomb(state, position))
        crate_hit_if_bomb = float(_crate_hit_if_bomb(state["field"], position))

    # escape_correct_move: only meaningful when in_escape_window holds --
    # does this action match the first step of the verified shortest path
    # to safety (spanning both the ticking-bomb phase and the lingering-
    # explosion phase, via state["escape_direction"]). BOMB is never a
    # valid action while in_escape_window is True (bombs_left only refills
    # one full step after the window has already closed), so there is no
    # BOMB special case here -- it's simply always 0.0, same as WAIT.
    if in_escape_window and action in DIRECTIONS:
        escape_correct_move = 1.0 if state["escape_direction"] == action else 0.0
    else:
        escape_correct_move = 0.0

    return np.array(
        [
            moves_to_coin,
            float(valid),
            1.0 if action == "WAIT" else 0.0,
            1.0 if action == "BOMB" else 0.0,
            moves_into_avoidable_danger,
            escape_exists_if_bomb,
            crate_hit_if_bomb,
            escape_correct_move,
            moves_to_crate,
        ]
    )


def _bfs_distance_map(field, sources):
    """
    Multi-source BFS distance map (e.g. distance to the nearest coin) over
    the free tiles of `field`. Unreachable tiles are left at np.inf.
    """
    distance_map = np.full(field.shape, np.inf)
    q = deque()  # double ended queue

    for src in sources:
        if field[src] == 0:
            distance_map[src] = 0
            q.append(src)

    while q:
        x, y = q.popleft()
        for dx, dy in DIRECTIONS.values():
            nx, ny = x + dx, y + dy

            if (
                0 <= nx < field.shape[0]  # width
                and 0 <= ny < field.shape[1]  # height
                and field[nx, ny] == 0  # empty field
                and distance_map[nx, ny] == np.inf  # untouched
            ):
                distance_map[nx, ny] = distance_map[x, y] + 1
                q.append((nx, ny))
    return distance_map


def _crate_adjacent_tiles(field):
    """
    Every free tile orthogonally adjacent to at least one crate -- the
    valid BFS *sources* for a "distance to nearest crate" map. Crates
    themselves are impassable (like stone walls, see _is_valid), so
    "distance to a crate" has to mean distance to a tile the agent could
    actually stand on next to it, not the crate's own coordinate.
    """
    width, height = field.shape
    tiles = []
    for x in range(width):
        for y in range(height):
            if field[x, y] != 0:
                continue
            for dx, dy in DIRECTIONS.values():
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and field[nx, ny] == 1:
                    tiles.append((x, y))
                    break
    return tiles


def _is_valid(state, action):
    """True if `action` is legal from the current state."""
    field = state["field"]
    position = state["position"]
    bomb_available = state["bomb_available"]
    occupied = state["occupied"]

    if action == "WAIT":
        return True
    if action == "BOMB":
        return bomb_available
    if action in DIRECTIONS:
        nx, ny = _simulate_move(position, action)

        # not within field
        if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
            return False

        # tile not free
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
      - danger_now: tiles that are lethal *this step* if occupied, i.e.
        tiles with an active explosion, plus the blast of any bomb whose
        timer has already reached 0 and will explode after this step's
        moves resolve.
      - bomb_blasts: for every bomb still ticking (timer > 0), the tiles
        its blast will eventually cover, so future-danger checks don't
        need to re-simulate blast propagation per action.
      - bomb_timers: the same ticking bombs' observed countdown values, so
        escape-path planning can use each bomb's true REMAINING budget
        (timer + 1 actions) instead of a flat constant.
    """
    field = game_state["field"]
    explosion_map = game_state["explosion_map"]
    danger_now = {
        (x, y) for x, y in zip(*np.nonzero(explosion_map > 0))
    }
    # remaining action budget for each currently-dangerous tile. This does
    # NOT mirror the bomb timer+1 formula: game_state["bombs"]'s timer is
    # the raw, unadjusted internal countdown, but explosion_map's value is
    # already (internal_timer - 1), so adding +1 here would just cancel
    # that offset back out. An explosion created with internal timer J,
    # observed at step i after creation, shows explosion_map == J - i and
    # is dangerous for steps i = 1..J-1, so the remaining dangerous steps
    # (including step i) is exactly the observed value, no adjustment. A
    # bomb whose timer has JUST hit 0 has exactly one action left,
    # overridden below.
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


def _escape_exists_after_bomb(state, position):
    """
    Simulates dropping a new bomb at `position` and runs a breadth-first
    search, bounded to ESCAPE_BUDGET steps, over free tiles (walls and
    crates both block movement) to check whether at least one tile
    *outside* the new bomb's blast is reachable in time. Tiles that are
    already dangerous (existing explosions / other ticking bombs) are
    excluded from the search, since walking through them would be fatal on
    the way out.
    """
    field = state["field"]
    new_blast = set(_simulate_blast_coords(position, field))
    forbidden = set(state["danger_now"])
    for blast in state["bomb_blasts"].values():
        forbidden.update(blast)

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
            if neighbor not in new_blast:
                return True
            frontier.append(neighbor)
    return False


def _bfs_first_step_out_of_blast(field, start, blast, forbidden, budget, target_distance_map=None):
    """
    BFS from `start`, avoiding `forbidden` tiles, over free tiles (walls and
    crates block movement, matching _escape_exists_after_bomb), bounded to
    `budget` moves. Returns (DIRECTION KEY, distance) for the first move
    taken along the shortest path to a tile outside `blast` and that tile's
    distance from `start`, or (None, None) if no such path exists within
    budget. Unlike _escape_exists_after_bomb, this reconstructs *which*
    first move the shortest path actually starts with (and how long it
    is), not just whether one exists -- the distance is surfaced so
    escape_potential can shape reward on it exactly like coin_potential
    does on coin_distance_map, without a second BFS.

    target_distance_map, if given, breaks ties among multiple equally-short
    escape routes by preferring whichever safe tile is closest to the
    current navigation target (coin_distance_map if a coin is visible,
    else crate_distance_map), rather than an arbitrary tile picked purely
    by DIRECTIONS' fixed iteration order. This meaningfully shortens the
    average time to fully clear a board, since it avoids the agent
    retreating to a safe tile that then requires backtracking afterward.
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
    field, position, danger_now, bomb_blasts, bomb_timers, danger_now_budget, target_distance_map=None
):
    """
    If the agent is currently threatened, find the shortest path to safety
    and return (direction of its first step, distance to safety) -- or
    (None, None) if unthreatened, or no escape exists within the true
    remaining budget. The distance is surfaced (not just the direction) so
    escape_potential can use it as a potential function exactly like
    coin_potential uses coin_distance_map. Two distinct threat sources are
    checked, covering the full danger window:
      1. an existing ticking bomb (via bomb_timers), using its true
         remaining timer+1 as budget -- the pre-detonation phase.
      2. the danger_now region itself (an already-detonated, still-lingering
         explosion, or a bomb at timer==0), using danger_now_budget the same
         way -- the post-detonation phase.
    Task 2 has at most one threat active at a time (bomb_available only
    resets once a bomb's danger fully clears), so at most one of the two
    checks below ever fires in practice.
    """
    for bomb_pos, timer in bomb_timers.items():
        blast = bomb_blasts[bomb_pos]
        if position not in blast:
            continue
        remaining_budget = timer + 1
        forbidden = set(danger_now)
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
        # forbidden must NOT include danger_now itself here -- we're
        # standing inside it and need to traverse through more of it to get
        # out, exactly as the ticking-bomb case never forbids its own blast
        forbidden = set()
        for blast in bomb_blasts.values():
            forbidden.update(blast)
        first_step, distance = _bfs_first_step_out_of_blast(
            field, position, set(danger_now), forbidden, remaining_budget, target_distance_map
        )
        if first_step is not None:
            return first_step, distance

    return None, None


def _crate_hit_if_bomb(field, position):
    """True if a bomb dropped at `position` would destroy at least one crate."""
    blast = _simulate_blast_coords(position, field)
    return any(field[x, y] == 1 for x, y in blast)


def coin_potential(state):
    """
    Potential-based shaping term: negative BFS distance to the nearest
    visible coin (Ng, Harada & Russell 1999 -- provably policy-invariant).
    """
    if state is None or state["coin_distance_map"] is None:
        return 0.0

    distance = state["coin_distance_map"][state["position"]]

    if np.isfinite(distance):
        return -distance
    else:
        return 0.0


def crate_potential(state):
    """
    Potential-based shaping term for crate-seeking: negative BFS distance
    to the nearest crate-adjacent tile. Mirrors coin_potential exactly, but
    reads from crate_distance_map and is gated to be active only when
    coin_potential is NOT (i.e. when no coin is currently visible) -- the
    same condition moves_to_crate is gated on, since this potential needs a
    feature that actually varies by direction for its shaped reward to
    attach to.
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


def q_values(weights, state):
    """Q(s, a) = weights . phi(s, a) for every action in ACTIONS."""
    values = np.zeros(len(ACTIONS))

    if state is None:
        return values

    for i, action in enumerate(ACTIONS):
        values[i] = np.dot(weights, state_action_features(state, action))

    return values
