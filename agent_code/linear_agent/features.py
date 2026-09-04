from collections import deque

import numpy as np

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]

# image coordinates, top left is (0,0)
DIRECTIONS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

N_FEATURES = 4


def state_features(game_state):
    """
    precomputes the parts indepedent of actions for the feature vector
    """

    if game_state is None:
        return None

    field = game_state["field"]  # 1 for crates, -1 for stone, 0 if free
    coins = game_state["coins"]  # coin coordinate list

    if coins:
        coin_distance_map = _bfs_distance_map(field, coins)
    else:
        coin_distance_map = None

    occupied = set()
    for bomb_position, _ in game_state["bombs"]:
        occupied.add(bomb_position)
    for other in game_state["others"]:
        occupied.add(other[3])

    return {
        "field": field,
        "position": game_state["self"][3],
        "bomb_available": game_state["self"][2],
        "occupied": occupied,
        "coin_distance_map": coin_distance_map,
    }


def state_action_features(state, action):
    """
    return feature vector
    """
    if state is None:
        return np.zeros(N_FEATURES)

    valid = _is_valid(state, action)
    moves_to_coin = 0.0

    if valid and state["coin_distance_map"] is not None:
        position = state["position"]
        new_position = _simulate_move(position, action)

        # current distance from coin on coin distance map
        current_distance = state["coin_distance_map"][position]
        new_distance = state["coin_distance_map"][new_position]

        if new_distance < current_distance:
            moves_to_coin = 1.0

    return np.array(
        [
            moves_to_coin,
            float(valid),
            1.0 if action == "WAIT" else 0.0,
            1.0 if action == "BOMB" else 0.0,
        ]
    )


def _bfs_distance_map(field, sources):
    """
    return distance map with multiple sources (here coins) using bfs
    """
    distance_map = np.full(field.shape, np.inf)
    q = deque()  # double ended queue

    for s in sources:
        if field[s] == 0:
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
                and distance_map[nx, ny] == np.inf  # untouched
            ):
                distance_map[nx, ny] = distance_map[x, y] + 1
                q.append((nx, ny))
    return distance_map


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
    potential function used for reward shaping, returns negative distance to coin.
    the reward is the discounted difference between the negative distances
    """
    if state is None or state["coin_distance_map"] is None:
        return 0.0

    distance = state["coin_distance_map"][state["position"]]

    if np.isfinite(distance):
        return -distance
    else:
        return 0.0
