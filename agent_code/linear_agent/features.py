from collections import deque

import numpy as np
from settings import BOMB_POWER, BOMB_TIMER

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]

# image coordinates, top left is (0,0)
DIRECTIONS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

FEATURE_NAMES = [
    "moves_to_coin",
    "moves_to_crate",
    "moves_to_safety",
    "moves_into_lethal",
    "moves_into_blast",
    "bias",  # constant
    "not_moving",  # invalid move or WAIT
    "bomb",
    "bomb_escapable",
    "bomb_hits",
]

N_FEATURES = len(FEATURE_NAMES)
GAMMA = 0.95  # discount factor

# scale for the potential rewards
COIN_POTENTIAL_SCALE = 0.25
CRATE_POTENTIAL_SCALE = 3.0


def state_features(game_state):
    """
    precomputes the parts indepedent of actions for the feature vector
    """

    if game_state is None:
        return None

    field = game_state["field"]  # 1 for crates, -1 for stone, 0 if free
    coins = game_state["coins"]  # coin coordinate list
    position = game_state["self"][3]

    others = {other[3] for other in game_state["others"]}

    occupied = set(others)
    for bomb_position, _ in game_state["bombs"]:
        occupied.add(bomb_position)

    # agents stands on his bomb after dropping
    blocked = occupied - {position}

    # currently lethal tiles + tiles that will be lethal
    lethal, blast = _danger_zones(game_state)

    if coins:
        coin_distance_map = _bfs_distance_map(field, coins, blocked)
    else:
        coin_distance_map = None

    return {
        "field": field,
        "position": position,
        "bomb_available": game_state["self"][2],
        "others": others,
        "occupied": occupied,
        "lethal": lethal,
        "blast": blast,
        "coin_distance_map": coin_distance_map,
        "crate_distance_map": _crate_distance_map(field, blocked),
        "safety_distance_map": _safety_distance_map(
            field, position, lethal, blast, blocked
        ),
    }


def state_action_features(state, action):
    """
    return feature vector
    """
    if state is None:
        return np.zeros(N_FEATURES)

    valid = _is_valid(state, action)
    position = state["position"]

    if valid:
        new_position = _simulate_move(position, action)
    else:
        new_position = position

    escapable = 0.0
    hits = 0.0
    if action == "BOMB" and valid:
        escapable = float(bomb_escapable(state))
        hits = float(bomb_hits(state))

    return np.array(
        [
            _moves_closer(state["coin_distance_map"], position, new_position),
            _moves_closer(state["crate_distance_map"], position, new_position),
            _moves_closer(state["safety_distance_map"], position, new_position),
            1.0 if new_position in state["lethal"] else 0.0,
            1.0 if new_position in state["blast"] else 0.0,
            1.0,  # bias
            1.0 if new_position == position and action != "BOMB" else 0.0,
            1.0 if action == "BOMB" else 0.0,
            escapable,
            hits,
        ]
    )


def _moves_closer(distance_map, position, new_position):
    """
    returns 1.0 if the new position is closer to the target on the distance map
    """
    if distance_map is None or new_position == position:
        return 0.0

    return 1.0 if distance_map[new_position] < distance_map[position] else 0.0


def _bfs_distance_map(field, sources, occupied=()):
    """
    return distance map for multiple sources (e.g. coins) using bfs.
    sources are seeded even when they are crates, which cannot be walked on
    """
    distance_map = np.full(field.shape, np.inf)
    q = deque()  # double ended queue

    for s in sources:
        distance_map[s] = 0
        q.append(s)

    while q:
        x, y = q.popleft()
        for dx, dy in DIRECTIONS.values():
            nx, ny = x + dx, y + dy

            if (
                0 <= nx < field.shape[0]  # width
                and 0 <= ny < field.shape[1]  # height
                and field[nx, ny] == 0  # empty field
                and (nx, ny) not in occupied  # bomb/agent positions
                and distance_map[nx, ny] == np.inf  # untouched
            ):
                distance_map[nx, ny] = distance_map[x, y] + 1
                q.append((nx, ny))
    return distance_map


def _crate_distance_map(field, occupied):
    """
    computes distance map to nearest tile next to crate (bombing purposes)
    """
    crates = []
    for crate in np.argwhere(field == 1):
        crates.append(tuple(crate))

    if not crates:
        return None

    return _bfs_distance_map(field, crates, occupied)


def _safety_distance_map(field, position, lethal, blast, occupied):
    """
    computes distance map to nearest tile outside an explosion/blast zone
    """
    if not lethal and not blast:
        return None

    safe = [
        tile
        for tile in map(tuple, np.argwhere(field == 0))
        if tile not in lethal and tile not in blast
    ]

    if not safe:
        return None

    return _bfs_distance_map(field, safe, occupied | (lethal - {position}))


def _danger_zones(game_state):
    """
    computes tiles that are currently lethal, and tiles that will be lethal
    """
    field = game_state["field"]

    lethal = set()
    for tile in np.argwhere(game_state["explosion_map"] > 0):
        lethal.add(tuple(tile))

    blast = set()

    for bomb_position, timer in game_state["bombs"]:
        if timer == 0:
            lethal.update(blast_coordinates(field, bomb_position))
        else:
            blast.update(blast_coordinates(field, bomb_position))

    return lethal, blast


def _is_valid(state, action):
    """
    checks if an action is valid
    """
    field = state["field"]
    position = state["position"]
    bomb_available = state["bomb_available"]
    occupied = state["occupied"]

    # check if action valid
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
    """
    return new position after action
    """
    if action in DIRECTIONS:
        dx, dy = DIRECTIONS[action]
        return (position[0] + dx, position[1] + dy)
    return position


def coin_potential(state):
    """
    used for reward shaping, returns negative distance to coin.
    the reward is the discounted difference between the negative distances
    """
    if state is None or state["coin_distance_map"] is None:
        return 0.0

    distance = state["coin_distance_map"][state["position"]]

    if np.isfinite(distance):
        return -COIN_POTENTIAL_SCALE * distance
    else:
        return 0.0


def escape_potential(state):
    """
    used for reward shaping, returns negative distance to the
    nearest safe tile
    """
    if state is None or state["safety_distance_map"] is None:
        return 0.0

    distance = state["safety_distance_map"][state["position"]]

    if np.isfinite(distance):
        return -distance
    else:
        return 0.0


def crate_potential(state):
    """
    used for reward shaping, rewards the step in which a bomb
    is dropped if it covers a crate
    """
    if state is None:
        return 0.0

    field = state["field"]

    for tile in state["blast"]:
        if field[tile] == 1:
            return CRATE_POTENTIAL_SCALE
    return 0.0


def blast_coordinates(field, bomb_position):
    """
    returns tiles hit by a bomb
    """
    x, y = bomb_position
    coordinates = [(x, y)]

    for dx, dy in DIRECTIONS.values():
        for i in range(1, BOMB_POWER + 1):
            nx, ny = x + i * dx, y + i * dy
            if field[nx, ny] == -1:  # stone wall
                break
            coordinates.append((nx, ny))
    return coordinates


def bomb_escapable(state):
    """
    checks if the a bomb at current position can be outrun
    """
    field = state["field"]
    position = state["position"]

    # distance map from current position
    distance_map = _bfs_distance_map(field, [position], occupied=state["occupied"])

    # mark all tiles within blast, from bomb at current position
    for tile in blast_coordinates(field, position):
        distance_map[tile] = np.inf

    return bool(np.any(distance_map <= BOMB_TIMER))


def bomb_hits(state):
    """
    returns true if a dropped bomb would hit a crate or an enemy agent
    """
    field = state["field"]

    for tile in blast_coordinates(field, state["position"]):
        if field[tile] == 1 or tile in state["others"]:
            return True
    return False


def q_values(weights, state):
    """
    computes q values for every action: Q(s,a) = weights * feature_vector
    """
    values = np.zeros(len(ACTIONS))

    if state is None:
        return values

    for i, action in enumerate(ACTIONS):
        values[i] = np.dot(weights, state_action_features(state, action))

    return values
