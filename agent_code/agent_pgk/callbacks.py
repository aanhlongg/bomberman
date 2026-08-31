import os
import pickle
import random
from collections import deque

import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Direction vectors in (dx, dy) image coordinates, matching the order of
# the first 4 entries of ACTIONS. Used both for feature encoding and for
# interpreting which move corresponds to which array index.
DIRECTION_VECTORS = {
    'UP': (0, -1),
    'RIGHT': (1, 0),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
}
VECTOR_TO_INDEX = {v: i for i, v in enumerate(DIRECTION_VECTORS.values())}

FEATURE_DIM = 8  # 4 (coin direction one-hot) + 4 (walkable neighbours)


def setup(self):
    """
    Called once before the first round. Initializes self.model, a weight
    matrix of shape (len(ACTIONS), FEATURE_DIM) used for a linear
    Q-function approximation: Q(s, a) = model[a] . features(s).
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = np.zeros((len(ACTIONS), FEATURE_DIM))
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    features = state_to_features(game_state)

    # 1. First, define the walkability mask so we know which moves are safe/valid
    valid_mask = np.ones(len(ACTIONS), dtype=bool)
    valid_mask[:4] = features[4:8] == 1.0   # UP, RIGHT, DOWN, LEFT
    valid_mask[4] = True                     # WAIT is always allowed
    valid_mask[5] = False                    # BOMB (disable unless implemented)

    # 2. Random exploration step (Epsilon-Greedy) with safety masking applied
    # Note: Make sure self.epsilon is initialized in setup() or setup_training()
    epsilon = getattr(self, 'epsilon', 0.1)
    if self.train and random.random() < epsilon:
        self.logger.debug("Choosing action randomly (masked).")
        valid_actions = [a for a, valid in zip(ACTIONS[:5], valid_mask[:5]) if valid]
        return np.random.choice(valid_actions)

    # 3. Exploitation step: Query model Q-values
    self.logger.debug("Querying model for action.")
    q_values = self.model @ features
    q_values_masked = np.where(valid_mask, q_values, -np.inf)

    # Break ties randomly among the best valid actions
    best_q = np.max(q_values_masked)
    best_actions = [a for a, q, ok in zip(ACTIONS, q_values_masked, valid_mask) if ok and q == best_q]
    return np.random.choice(best_actions)

def _bfs_first_step_towards_nearest_coin(field: np.array, start: tuple, coins: list):
    """
    Breadth-first search from `start` over free tiles (field == 0) to the
    nearest coin. Returns the coordinate of the first step to take, or
    None if no coin is reachable (or there are no coins).
    """
    if not coins:
        return None

    if start in coins:
        # Already standing on a coin tile -- no direction to move, it will
        # be collected regardless of this step's action.
        return None

    coin_set = set(coins)
    visited = {start: None}
    queue = deque([start])

    while queue:
        current = queue.popleft()
        if current in coin_set:
            # Walk the parent chain back to find the first step from `start`.
            step = current
            while visited[step] != start:
                step = visited[step]
            return step

        x, y = current
        for dx, dy in DIRECTION_VECTORS.values():
            neighbour = (x + dx, y + dy)
            nx, ny = neighbour
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                continue
            if field[nx, ny] != 0:  # blocked by wall or crate
                continue
            if neighbour in visited:
                continue
            visited[neighbour] = current
            queue.append(neighbour)

    return None  # no coin reachable


def state_to_features(game_state: dict) -> np.array:
    """
    Builds an 8-dim feature vector:
      [0:4] one-hot direction (UP, RIGHT, DOWN, LEFT) towards the nearest
            reachable coin, all-zero if no coin is reachable.
      [4:8] whether the tile in that same direction (UP, RIGHT, DOWN, LEFT)
            is currently walkable (free tile).
    """
    if game_state is None:
        return None

    field = game_state['field']
    _, _, _, (x, y) = game_state['self']
    coins = game_state['coins']

    coin_features = np.zeros(4)
    step = _bfs_first_step_towards_nearest_coin(field, (x, y), coins)
    if step is not None:
        vector = (step[0] - x, step[1] - y)
        idx = VECTOR_TO_INDEX.get(vector)
        if idx is not None:
            coin_features[idx] = 1.0

    wall_features = np.zeros(4)
    for i, (dx, dy) in enumerate(DIRECTION_VECTORS.values()):
        nx, ny = x + dx, y + dy
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1] and field[nx, ny] == 0:
            wall_features[i] = 1.0

    return np.concatenate([coin_features, wall_features])