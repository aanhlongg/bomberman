"""
State representation and the linear Q-function for the linear_agent.

The agent is a linear function approximator: Q(s, a) = weights . phi(s, a),
where phi(s, a) is the N_FEATURES-dimensional feature vector state_action_features()
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
# 0: moves_to_coin          6: crate_hit_if_bomb
# 1: valid                  7: escape_correct_move
# 2: is_wait                8: moves_to_crate
# 3: is_bomb                9: moves_to_opponent
# 4: moves_into_avoidable_danger   10: bomb_hits_opponent_if_bomb
# 5: escape_exists_if_bomb
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
#
# 9 (moves_to_opponent) is deliberately NOT mutually exclusive with 0/8:
# hunting an opponent (Task 3) is an independent objective layered on top of
# coin/crate seeking, not a replacement for it, so it can be simultaneously
# nonzero alongside whichever of 0/8 is currently active. Reward magnitude,
# not feature gating, is what should make hunting dominate when it matters.
# Gating it to 0 during in_escape_window (mirroring 0/8) has been tried
# TWICE and reverted twice -- both times it collapsed escape_correct_move
# into a structural tie (every direction equally 0-valued) rather than
# letting it develop real signal. See state_action_features for the full
# writeup.
#
# A graded version of escape_exists_if_bomb (distinguishing "route exists
# with slack" from "route exists but uses the full ESCAPE_BUDGET with zero
# steps to spare") was tried at this index and reverted: it's collinear
# with escape_exists_if_bomb by construction (having margin implies a
# route exists, so one is always a subset of the other), and splitting the
# BOMB decision's signal across two correlated features diluted both --
# kills dropped (28->19) and suicides rose (21->32) rather than the
# intended improvement. The underlying idea (bomb placement should weigh
# escape margin, not just existence) may still be worth revisiting in a
# non-collinear form.
#
# Splitting is_wait into is_wait / is_wait_while_chaseable (gated on a new
# chase_opportunity_exists state flag -- is there a valid direction that
# reduces distance to the opponent right now) was tried at this index and
# reverted. The theory: a single shared is_wait weight can't distinguish
# "nothing else to do, waiting is fine" from "an opponent is chaseable and
# waiting squanders it," and credit from the far more common ordinary
# contexts was inflating w_wait past w_moves_to_opponent, causing the
# agent to prefer idling over hunting once a board was fully cleared
# (confirmed: 83.5% kills either way, but a direct trace of 12 real
# cleared-board moments showed WAIT chosen in every single one both
# before and after). The split made it WORSE, not better --
# w_is_wait_while_chaseable settled even higher (~9.2) than the original
# undifferentiated w_wait (~7.6) it replaced, confirmed by re-tracing the
# same cleared-board scenario post-fix. Likely cause: opponent_potential's
# shaping reflects the OPPONENT's own movement, which for a
# randomly-wandering opponent happens independent of our action -- WAIT-ing
# while they happen to wander closer still earns positive shaped reward,
# so credit accumulates on is_wait_while_chaseable specifically regardless
# of what we actually did. Splitting the feature gave that
# misattribution problem a dedicated weight to inflate rather than fixing
# the underlying credit-assignment issue.
N_FEATURES = 11
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

    opponent_positions = [other[3] for other in game_state["others"]]
    # (position, bomb_available) per opponent -- used by
    # _escape_exists_after_bomb to account for a possible retaliatory bomb
    # from an adjacent, bomb-capable opponent (see its docstring).
    opponents_with_bomb = [(other[3], other[2]) for other in game_state["others"]]
    # multi-source BFS to the nearest opponent, mirroring coin_distance_map
    # exactly -- always computed (opponents, unlike coins, are never hidden
    # under a crate, so there's no analogous "nothing to seek" case), and
    # independent of whether a coin/crate is also being pursued this step
    opponent_distance_map = (
        _bfs_distance_map(field, opponent_positions) if opponent_positions else None
    )

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
        field, position, danger_now, bomb_blasts, bomb_timers, danger_now_budget,
        target_distance_map, opponent_positions,
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
        "opponent_distance_map": opponent_distance_map,
        "opponent_positions": opponent_positions,
        "opponents_with_bomb": opponents_with_bomb,
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
    bomb_hits_opponent_if_bomb = 0.0
    if action == "BOMB" and valid:
        escape_exists_if_bomb = float(_escape_exists_after_bomb(state, position))
        crate_hit_if_bomb = float(_crate_hit_if_bomb(state["field"], position))
        # MVP proxy: does the blast cover an opponent's CURRENT tile. Does
        # not account for the opponent moving before the bomb detonates
        # (ESCAPE_BUDGET steps later) -- a real limitation for a moving
        # target. A stricter version requiring the opponent to have no
        # escape route of their own (mirroring our own
        # _escape_exists_after_bomb from their position) was tried and
        # reverted: w_bomb_hits_opponent learned a strongly NEGATIVE weight
        # (down to -6.66) instead of the intended positive one, and both
        # kills (28->22) and suicides (21->40, nearly doubling) got worse.
        # The stricter check likely over-reports "cornered" in some cases,
        # teaching the model that the bonus-paying signal doesn't reliably
        # predict a real kill and pulling the agent into riskier bombing.
        # Root cause not fully diagnosed; reverted to the simpler proxy
        # rather than debug further given the regression's severity.
        blast = set(_simulate_blast_coords(position, state["field"]))
        bomb_hits_opponent_if_bomb = float(
            any(opp in blast for opp in state["opponent_positions"])
        )

    # moves_to_opponent: mirrors moves_to_coin, but NOT gated to be mutually
    # exclusive with moves_to_coin/moves_to_crate (see N_FEATURES comment) --
    # hunting is an independent objective, not a replacement for the
    # existing ones.
    #
    # Zeroing this during in_escape_window (mirroring moves_to_coin) has now
    # been tried TWICE and reverted twice. First attempt (pre-seed-sweep):
    # escape_correct_move was stuck near 0.0, so removing moves_to_opponent
    # left no differentiating signal at all -- total ties, 100% self-kill.
    # Second attempt (after trace analysis showed escape_correct_move
    # reliably reaching 0.6-1.5 and losing the argmax to moves_to_opponent's
    # 3.6-4.4): reasoned this meant the competing pull was now purely
    # harmful and safe to remove. Instead, removing it caused an even more
    # severe collapse (200/200 self-kills, w_escape_correct_move pinned
    # within +-0.3 of zero for all 5,000 rounds). The lesson: with
    # moves_to_coin/moves_to_crate already zeroed during escape windows,
    # moves_to_opponent zeroed too leaves escape_correct_move as the ONLY
    # action-varying feature there -- at its zero initialization every
    # direction ties, argmax always picks the same one regardless of
    # correctness, and that structural tie apparently prevented the weight
    # from ever developing real signal, unlike when moves_to_opponent's
    # competing pull was present to create actual reward variance between
    # actions. escape_correct_move's earlier 0.6-1.5 development turns out
    # to have DEPENDED on that competition, not merely survived it. Left
    # active during escape; the fix for escape losing the argmax needs to
    # come from elsewhere (e.g. strengthening escape's own signal without
    # creating a tie), not from removing the only other one.
    moves_to_opponent = 0.0
    if valid and state["opponent_distance_map"] is not None:
        current_distance = state["opponent_distance_map"][position]
        new_distance = state["opponent_distance_map"][new_position]

        if new_distance < current_distance:
            moves_to_opponent = 1.0

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
            moves_to_opponent,
            bomb_hits_opponent_if_bomb,
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
    *outside* the new bomb's blast (and any retaliatory blast, see below)
    is reachable in time. Tiles that are already dangerous (existing
    explosions / other ticking bombs) are excluded from the search, since
    walking through them would be fatal on the way out -- as are tiles
    currently occupied by an opponent (Task 3): an opponent's exact
    position can change before the agent gets there, but treating it as
    free right now would let this search "find" an escape through a tile
    that's actually blocked at this exact moment, which _is_valid would
    then reject.

    Also accounts for a possible RETALIATORY bomb from any opponent
    currently adjacent to `position` (Manhattan distance <= 1) with a bomb
    available. Trace analysis vs rule_based_agent showed it reliably drops
    a bomb back at its own current tile whenever we bomb it while
    adjacent (its own rule-based logic checks exactly this condition) --
    two simultaneous blasts from adjacent tiles routinely leave no escape
    within the fixed budget, but this feature previously only ever
    accounted for our own blast, so it looked "safe" right up until the
    opponent's own bomb closed the last exit at the same instant.
    peaceful_agent and coin_collector_agent never bomb an adjacent
    opponent deliberately, so this is a no-op against them in practice
    (opponents_with_bomb is empty or none are ever adjacent+bomb-capable
    at decision time often enough to matter) -- it's specifically aimed at
    opponents that do.
    """
    field = state["field"]
    new_blast = set(_simulate_blast_coords(position, field))
    forbidden = set(state["danger_now"]) | set(state["opponent_positions"])
    for blast in state["bomb_blasts"].values():
        forbidden.update(blast)

    retaliation_blast = set()
    for opp_pos, opp_bomb_available in state["opponents_with_bomb"]:
        if opp_bomb_available and abs(opp_pos[0] - position[0]) + abs(opp_pos[1] - position[1]) <= 1:
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
    field, position, danger_now, bomb_blasts, bomb_timers, danger_now_budget,
    target_distance_map=None, opponent_positions=(),
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

    opponent_positions (Task 3): treated as forbidden in both branches, for
    the same reason _escape_exists_after_bomb does -- an opponent occupies
    its current tile right now, so a route planned through it isn't
    actually executable this step, even though the opponent may well have
    moved off it by the time the agent would arrive.
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
        # forbidden must NOT include danger_now itself here -- we're
        # standing inside it and need to traverse through more of it to get
        # out, exactly as the ticking-bomb case never forbids its own blast
        forbidden = set(opponent_positions)
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


# Scales opponent_potential's raw BFS-distance contribution relative to
# coin_potential/crate_potential's implicit scale of 1.0. A positive
# scalar multiple of a valid potential function is itself still a valid
# (policy-invariant) potential function.
#
# Tried at 0.3 (shrinking it) on the theory that w_moves_to_opponent
# growing faster than w_escape_correct_move was crowding out escape
# priority during escape windows -- reverted: w_escape_correct_move stayed
# stuck near 0.0 regardless (100% self-kill either way), showing it wasn't
# actually competing for gradient with moves_to_opponent at all.
# moves_to_opponent's large, unscaled weight was apparently acting as an
# accidental substitute escape heuristic (favoring movement away from the
# opponent, which correlates with safety often enough to matter);
# shrinking it just removed that crutch without fixing the real problem,
# which is that escape_correct_move isn't developing properly in the
# Task 3 setting for a still-unknown reason. Left at 1.0 (no scaling)
# pending that investigation.
OPPONENT_POTENTIAL_SCALE = 1.0


def opponent_potential(state):
    """
    Potential-based shaping term (Task 3): negative BFS distance to the
    nearest opponent, scaled by OPPONENT_POTENTIAL_SCALE. Unlike
    coin_potential/crate_potential, always active when at least one
    opponent exists -- hunting is layered on top of coin/crate seeking,
    not gated to be exclusive with it (see the N_FEATURES comment on
    moves_to_opponent).
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
