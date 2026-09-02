import os
import pickle
import random
from collections import deque

import numpy as np

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]

DIRECTION_VECTORS = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
}
VECTOR_TO_INDEX = {v: i for i, v in enumerate(DIRECTION_VECTORS.values())}

FEATURE_DIM = 10  # 4 (coin direction) + 4 (walkable) + 1 (bias) + 1 (distance)

# Set EVAL_MODE=1 in the environment to force purely greedy (epsilon=0)
# action selection with no weight updates -- used by evaluate.py's --eval flag.
EVAL_MODE = os.environ.get("EVAL_MODE") == "1"

# Optional: set MODEL_TAG to keep separate runs from overwriting each
# other's model/stats files. Leave unset for the plain default filenames.
MODEL_TAG = os.environ.get("MODEL_TAG", "")


def _model_filename() -> str:
    return f"my-saved-model{'_' + MODEL_TAG if MODEL_TAG else ''}.pt"


def setup(self):
    self.current_coin_target = None  # used by sticky coin-targeting in state_to_features

    model_file = _model_filename()
    if os.path.isfile(model_file):
        self.logger.info(f"Loading model from saved state ({model_file}).")
        with open(model_file, "rb") as file:
            self.model = pickle.load(file)
    else:
        self.logger.info(f"Setting up model from scratch ({model_file}).")
        self.model = np.random.normal(0, 0.01, size=(len(ACTIONS), FEATURE_DIM))


def act(self, game_state: dict) -> str:
    features = state_to_features(self, game_state)

    # Valid move mask: check walkability from features[4:8]
    valid_mask = np.ones(len(ACTIONS), dtype=bool)
    valid_mask[:4] = features[4:8] == 1.0  # UP, RIGHT, DOWN, LEFT
    valid_mask[4] = True   # WAIT allowed (this MUST match the bootstrap mask in train.py's _update_q)
    valid_mask[5] = False  # BOMB disabled -- not needed for this task

    if self.train and not EVAL_MODE:
        ep = getattr(self, "episode_num", 1)
        epsilon = max(0.05, 0.5 * (0.99**ep))
    else:
        epsilon = 0.0

    if self.train and not EVAL_MODE and random.random() < epsilon:
        valid_indices = np.where(valid_mask)[0]
        return ACTIONS[np.random.choice(valid_indices)]

    q_values = self.model @ features
    q_values[~valid_mask] = -1e9

    max_q = np.max(q_values)
    best_actions = np.where(q_values == max_q)[0]

    return ACTIONS[np.random.choice(best_actions)]


def _get_blocked_tiles(game_state: dict) -> set:
    """
    Tiles blocked by other agents or bombs. Always empty for this task
    (no opponents, no bombs), kept for robustness / future reuse.
    """
    blocked = set()
    for _, _, _, pos in game_state.get("others", []):
        blocked.add(pos)
    for bomb_pos, _timer in game_state.get("bombs", []):
        blocked.add(bomb_pos)
    return blocked


def _bfs_nearest_coin(field: np.array, start: tuple, coins: list, blocked: set = frozenset()):
    """
    BFS from `start` to the nearest coin. Returns (coin_position, first_step,
    distance), or (None, None, None) if there are no coins or none reachable.
    """
    if not coins:
        return None, None, None

    coin_set = set(coins)
    if start in coin_set:
        return start, None, 0

    visited = {start: None}
    depths = {start: 0}
    queue = deque([start])

    while queue:
        current = queue.popleft()
        if current in coin_set:
            step = current
            while visited[step] != start:
                step = visited[step]
            return current, step, depths[current]

        x, y = current
        for dx, dy in DIRECTION_VECTORS.values():
            n = (x + dx, y + dy)
            nx, ny = n
            if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
                if field[nx, ny] == 0 and n not in blocked and n not in visited:
                    visited[n] = current
                    depths[n] = depths[current] + 1
                    queue.append(n)

    return None, None, None


def _bfs_distance_and_step(field: np.array, start: tuple, target: tuple, blocked: set = frozenset()):
    """
    BFS from `start` to a specific `target` tile. Returns (first_step,
    distance), or (None, None) if unreachable.
    """
    if target is None:
        return None, None
    if start == target:
        return None, 0

    visited = {start: None}
    depths = {start: 0}
    queue = deque([start])

    while queue:
        current = queue.popleft()
        if current == target:
            step = current
            while visited[step] != start:
                step = visited[step]
            return step, depths[current]

        x, y = current
        for dx, dy in DIRECTION_VECTORS.values():
            n = (x + dx, y + dy)
            nx, ny = n
            if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
                if field[nx, ny] == 0 and n not in blocked and n not in visited:
                    visited[n] = current
                    depths[n] = depths[current] + 1
                    queue.append(n)

    return None, None


# Only switch away from the currently-targeted coin if an alternative is
# at least this many tiles closer -- prevents flip-flopping between two
# similarly-distant coins on consecutive steps.
STICKY_MARGIN = 2


def state_to_features(self, game_state: dict) -> np.array:
    if game_state is None:
        return None

    field = game_state["field"]
    _, _, _, (x, y) = game_state["self"]
    coins = game_state["coins"]
    coin_set = set(coins)
    blocked = _get_blocked_tiles(game_state)

    # Sticky coin targeting: keep chasing the same coin across steps unless
    # it's been collected, or a different coin is meaningfully closer.
    sticky_target = getattr(self, "current_coin_target", None)
    if sticky_target is not None and sticky_target not in coin_set:
        sticky_target = None

    nearest_coin, nearest_step, nearest_dist = _bfs_nearest_coin(field, (x, y), coins, blocked)

    if sticky_target is not None and sticky_target != nearest_coin:
        sticky_step, sticky_dist = _bfs_distance_and_step(field, (x, y), sticky_target, blocked)
        if sticky_dist is not None and (nearest_dist is None or nearest_dist >= sticky_dist - STICKY_MARGIN):
            target, next_step = sticky_target, sticky_step
        else:
            target, next_step = nearest_coin, nearest_step
    else:
        target, next_step = nearest_coin, nearest_step

    self.current_coin_target = target

    # 1. Direction indicator (4 dims)
    coin_features = np.zeros(4)
    if next_step is not None:
        vector = (next_step[0] - x, next_step[1] - y)
        idx = VECTOR_TO_INDEX.get(vector)
        if idx is not None:
            coin_features[idx] = 1.0

    # 2. Walkability (4 dims)
    walkable_features = np.zeros(4)
    for i, (dx, dy) in enumerate(DIRECTION_VECTORS.values()):
        nx, ny = x + dx, y + dy
        if (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]
                and field[nx, ny] == 0 and (nx, ny) not in blocked):
            walkable_features[i] = 1.0

    # 3. Distance to targeted coin (1 dim)
    dist_feature = np.array([0.0])
    if coins:
        ref_dist = nearest_dist if target == nearest_coin else _bfs_distance_and_step(field, (x, y), target, blocked)[1]
        if ref_dist is not None:
            dist_feature[0] = ref_dist / 30.0

    bias_feature = np.array([1.0])

    return np.concatenate(
        [coin_features, walkable_features, dist_feature, bias_feature]
    )