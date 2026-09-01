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


def setup(self):
    model_path = os.path.join(os.path.dirname(__file__), "my-saved-model.pt")
    if os.path.isfile("my-saved-model.pt"):
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
    else:
        self.logger.info("Setting up model from scratch.")
        # Small random weights close to zero
        self.model = np.random.normal(0, 0.01, size=(len(ACTIONS), FEATURE_DIM))


def act(self, game_state: dict) -> str:
    features = state_to_features(game_state)

    # Valid move mask: check walkability from features[4:8]
    valid_mask = np.ones(len(ACTIONS), dtype=bool)
    valid_mask[:4] = features[4:8] == 1.0  # UP, RIGHT, DOWN, LEFT
    valid_mask[4] = False  # WAIT always allowed
    valid_mask[5] = False  # BOMB disabled for coin collector

    # Dynamic Epsilon decay
    if self.train:
        ep = getattr(self, "episode_num", 1)
        epsilon = max(0.05, 0.5 * (0.99**ep))
    else:
        epsilon = 0.0

    # Exploration
    if self.train and random.random() < epsilon:
        valid_indices = np.where(valid_mask)[0]
        return ACTIONS[np.random.choice(valid_indices)]

    # Exploitation: Q = W * x
    q_values = self.model @ features
    q_values[~valid_mask] = -1e9  # Mask invalid moves

    # Pick best action among valid ones
    max_q = np.max(q_values)
    best_actions = np.where(q_values == max_q)[0]

    return ACTIONS[np.random.choice(best_actions)]


def _bfs_first_step_towards_nearest_coin(field: np.array, start: tuple, coins: list):
    """
    Standard BFS returning the exact coordinate of the next step toward the closest coin.
    """
    if not coins:
        return None

    coin_set = set(coins)
    visited = {start: None}
    queue = deque([start])

    while queue:
        current = queue.popleft()

        if current in coin_set and current != start:
            # Reconstruct first step from start
            step = current
            while visited[step] != start:
                step = visited[step]
            return step

        x, y = current
        for dx, dy in DIRECTION_VECTORS.values():
            neighbour = (x + dx, y + dy)
            nx, ny = neighbour
            if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
                if field[nx, ny] == 0 and neighbour not in visited:
                    visited[neighbour] = current
                    queue.append(neighbour)

    return None


def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        return None

    field = game_state["field"]
    _, _, _, (x, y) = game_state["self"]
    coins = game_state["coins"]

    # 1. Direction indicator from BFS (4 dims)
    coin_features = np.zeros(4)
    next_step = _bfs_first_step_towards_nearest_coin(field, (x, y), coins)
    if next_step is not None:
        vector = (next_step[0] - x, next_step[1] - y)
        idx = VECTOR_TO_INDEX.get(vector)
        if idx is not None:
            coin_features[idx] = 1.0

    # 2. Walkability (4 dims)
    walkable_features = np.zeros(4)
    for i, (dx, dy) in enumerate(DIRECTION_VECTORS.values()):
        nx, ny = x + dx, y + dy
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1] and field[nx, ny] == 0:
            walkable_features[i] = 1.0

    # 3. Distance Feature (1 dim) - normalized distance to nearest coin
    dist_feature = np.array([0.0])
    if coins:
        # Distance via BFS or Manhattan distance normalized to grid size (e.g. max 30)
        min_dist = min(abs(x - cx) + abs(y - cy) for cx, cy in coins)
        dist_feature[0] = min_dist / 30.0

    bias_feature = np.array([1.0])

    return np.concatenate(
        [coin_features, walkable_features, dist_feature, bias_feature]
    )
